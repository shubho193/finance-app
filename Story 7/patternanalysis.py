import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from typing import List, Dict
import json
from datetime import datetime
from typing import Dict, Any

# ====================== DATA CLASSES ======================
class Transaction:
    def __init__(self, date: str, amount: float, description: str):
        self.date = date
        self.amount = amount
        self.description = description
        self.category = self._categorize()
    
    def _categorize(self) -> str:
        categories = {
            "Food": ["mcdonalds", "starbucks", "groceries", "restaurant"],
            "Transport": ["uber", "lyft", "gas", "metro", "subway"],
            "Entertainment": ["netflix", "spotify", "cinema", "concert"]
        }
        desc_lower = self.description.lower()
        for category, keywords in categories.items():
            if any(kw in desc_lower for kw in keywords):
                return category
        return "Other"

class UserProfile:
    def __init__(self, user_id: str, age: int, income: float, transactions: List[Transaction]):
        self.user_id = user_id
        self.age = age
        self.income = income
        self.transactions = transactions
    
    def get_spending_by_category(self) -> Dict[str, float]:
        """Calculate total spending per category"""
        spending = {}
        for txn in self.transactions:
            spending[txn.category] = spending.get(txn.category, 0) + txn.amount
        return spending

# ====================== ANALYSIS COMPONENTS ======================
class PeerAnalyzer:
    def __init__(self, peer_profiles: List[UserProfile]):
        self.peer_profiles = peer_profiles
        self.kmeans_model = None
    
    def _prepare_features(self) -> np.ndarray:
        """Convert user profiles to feature matrix for clustering"""
        features = []
        for user in self.peer_profiles:
            spending = user.get_spending_by_category()
            features.append([
                user.age,
                user.income,
                spending.get("Food", 0),
                spending.get("Transport", 0),
                spending.get("Entertainment", 0)
            ])
        return np.array(features)
    
    def find_similar_users(self, target_user: UserProfile) -> List[UserProfile]:
        """Find users with similar spending patterns using K-Means"""
        features = self._prepare_features()
        
        # Use elbow method or set fixed clusters for simplicity
        self.kmeans_model = KMeans(n_clusters=3, random_state=42).fit(features)
        
        # Prepare target user's feature vector
        target_spending = target_user.get_spending_by_category()
        target_features = np.array([
            target_user.age,
            target_user.income,
            target_spending.get("Food", 0),
            target_spending.get("Transport", 0),
            target_spending.get("Entertainment", 0)
        ]).reshape(1, -1)
        
        # Find cluster members
        cluster_label = self.kmeans_model.predict(target_features)[0]
        return [
            user for i, user in enumerate(self.peer_profiles) 
            if self.kmeans_model.labels_[i] == cluster_label
        ]

class MarketAnalyzer:
    @staticmethod
    def fetch_inflation_rate() -> float:
        """Fetch current inflation rate from API (mock implementation)"""
        try:
            # Mock API response - in practice use World Bank/FRED API
            mock_response = {'inflation_rate': 6.5}
            return float(mock_response['inflation_rate'])
        except:
            return 3.0
    
    @staticmethod
    def adjust_for_inflation(values: Dict[str, float], inflation_rate: float) -> Dict[str, float]:
        """Adjust monetary values for inflation"""
        return {k: v * (1 + inflation_rate/100) for k, v in values.items()}

# ====================== MAIN ANALYSIS ENGINE ======================
class SpendingAnalyzer:
    def __init__(self, user: UserProfile, peer_profiles: List[UserProfile]):
        self.user = user
        self.peer_analyzer = PeerAnalyzer(peer_profiles)
        self.market_analyzer = MarketAnalyzer()
    
    def analyze(self) -> Dict:
        """Run complete analysis pipeline"""
        # 1. Get user's spending breakdown
        user_spending = self.user.get_spending_by_category()
        
        # 2. Find comparable peers
        similar_users = self.peer_analyzer.find_similar_users(self.user)
        peer_benchmark = self._calculate_average_spending(similar_users)
        
        # 3. Incorporate market trends
        inflation_rate = self.market_analyzer.fetch_inflation_rate()
        adjusted_benchmark = self.market_analyzer.adjust_for_inflation(peer_benchmark, inflation_rate)
        
        # 4. Generate insights
        insights = self._generate_insights(user_spending, peer_benchmark, adjusted_benchmark)
        
        return {
            "user_profile": {
                "age": self.user.age,
                "income": self.user.income,
                "spending": user_spending
            },
            "peer_comparison": {
                "similar_users_count": len(similar_users),
                "average_spending": peer_benchmark,
                "inflation_adjusted": adjusted_benchmark
            },
            "market_conditions": {
                "inflation_rate": inflation_rate
            },
            "insights": insights
        }
    
    def _calculate_average_spending(self, users: List[UserProfile]) -> Dict[str, float]:
        """Calculate average spending for a user group"""
        if not users:
            return {}
            
        totals = {}
        counts = {}
        for user in users:
            spending = user.get_spending_by_category()
            for category, amount in spending.items():
                totals[category] = totals.get(category, 0) + amount
                counts[category] = counts.get(category, 0) + 1
        return {k: round(v/counts[k], 2) for k, v in totals.items()}
    
    def _generate_insights(self, user_spending: Dict, peer_avg: Dict, adjusted_avg: Dict) -> List[str]:
        """Generate actionable insights"""
        insights = []
        
        for category in user_spending:
            if category not in peer_avg:
                continue
                
            user_amt = user_spending[category]
            peer_amt = peer_avg[category]
            adjusted_amt = adjusted_avg.get(category, peer_amt)
            
            diff = user_amt - adjusted_amt
            pct_diff = (diff / adjusted_amt) * 100 if adjusted_amt != 0 else 0
            
            if pct_diff > 15:
                insights.append(
                    f"You're spending {abs(pct_diff):.1f}% more on {category} "
                    f"than similar peers (${user_amt:.2f} vs ${adjusted_amt:.2f})"
                )
            elif pct_diff < -15:
                insights.append(
                    f"You're spending {abs(pct_diff):.1f}% less on {category} "
                    f"than similar peers (${user_amt:.2f} vs ${adjusted_amt:.2f})"
                )
        
        if not insights:
            insights.append("Your spending aligns well with similar peers")
            
        return insights
    
    def export_to_json(self, analysis_results: Dict[str, Any], filename: str = None) -> str:
        """
        Exports analysis results to JSON with timestamp.
        Structure includes:
        - User profile
        - Peer comparisons
        - Market trends
        - Generated insights
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spending_analysis_{timestamp}.json"
        
        # Prepare structured data
        export_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "analysis_version": "1.0"
            },
            "user_profile": analysis_results["user_profile"],
            "peer_comparison": analysis_results["peer_comparison"],
            "market_conditions": analysis_results["market_conditions"],
            "insights": analysis_results["insights"],
            "spending_by_category": {
                "user": analysis_results["user_profile"]["spending"],
                "peers": analysis_results["peer_comparison"]["average_spending"],
                "inflation_adjusted": analysis_results["peer_comparison"]["inflation_adjusted"]
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Analysis exported to {filename}")
        return filename

# ====================== EXAMPLE USAGE ======================
def create_sample_data() -> List[UserProfile]:
    """Generate mock data for testing"""
    profiles = []
    transactions = [
        # Student profile 1
        [("2023-01-01", 5.50, "Starbucks coffee"), 
         ("2023-01-02", 25.00, "Groceries"),
         ("2023-01-05", 15.00, "Uber ride")],
         
        # Student profile 2
        [("2023-01-01", 8.00, "Campus dining"), 
         ("2023-01-03", 12.00, "Spotify subscription"),
         ("2023-01-04", 20.00, "Gas station")],
         
        # Professional profile
        [("2023-01-01", 50.00, "Steakhouse"), 
         ("2023-01-02", 80.00, "Weekend groceries"),
         ("2023-01-06", 60.00, "Concert tickets")]
    ]
    
    profiles.append(UserProfile("student1", 20, 1500, [Transaction(*t) for t in transactions[0]]))
    profiles.append(UserProfile("student2", 22, 1800, [Transaction(*t) for t in transactions[1]]))
    profiles.append(UserProfile("professional", 35, 6000, [Transaction(*t) for t in transactions[2]]))
    
    return profiles

def main():
    # 1. Prepare data
    peers = create_sample_data()
    test_user = peers[0]  # Analyze first student
    
    # 2. Run analysis
    analyzer = SpendingAnalyzer(test_user, peers)
    results = analyzer.analyze()
    
    # 3. Display results
    print("\n=== Spending Analysis Report ===")
    print(f"User: {results['user_profile']['age']}yo, ${results['user_profile']['income']}/mo")
    print("\nSpending Breakdown:")
    for cat, amt in results['user_profile']['spending'].items():
        print(f"- {cat}: ${amt:.2f}")
    
    print("\nPeer Comparison:")
    print(f"Compared with {results['peer_comparison']['similar_users_count']} similar users")
    for cat, amt in results['peer_comparison']['average_spending'].items():
        print(f"- Avg {cat}: ${amt:.2f}")
    
    print(f"\nMarket Conditions (Inflation: {results['market_conditions']['inflation_rate']}%)")
    
    print("\nInsights:")
    for insight in results['insights']:
        print(f"- {insight}")
    
    # Run analysis and export
    analyzer = SpendingAnalyzer(test_user, peers)
    results = analyzer.analyze()
    
    # Export to JSON
    analyzer.export_to_json(results)  # Auto-generates filename
    # Or specify custom name: analyzer.export_to_json(results, "my_analysis.json")

if __name__ == "__main__":
    main()
