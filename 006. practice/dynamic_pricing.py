# Dynamic Pricing for Automobile Resale Industry
# Inspired by AUTO1 Group's operations
# Complete end-to-end implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# For SHAP interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

plt.style.use('seaborn-v0_8')
np.random.seed(42)

# ================================================================
# 1. PROBLEM UNDERSTANDING
# ================================================================

"""
WHY DYNAMIC PRICING IS CRUCIAL IN AUTO RESALE INDUSTRY:

AUTO1 Group operates as Europe's leading digital automotive platform, buying and selling
used cars through data-driven approaches. Dynamic pricing is critical because:

1. MARKET VOLATILITY: Car values fluctuate based on demand, seasonality, fuel prices
2. INVENTORY OPTIMIZATION: Quick turnover reduces holding costs and storage
3. COMPETITIVE ADVANTAGE: Real-time pricing beats static dealer models
4. PROFIT MAXIMIZATION: Optimal pricing balances quick sale vs. maximum profit
5. RISK MITIGATION: Prevents overpricing slow-moving inventory

AUTO1's business model benefits through:
- Faster inventory turnover (target: <30 days)
- Optimized profit margins per vehicle category
- Reduced price discovery time
- Automated decision-making at scale
"""

# ================================================================
# 2. DATA COLLECTION & SIMULATION
# ================================================================

class DataGenerator:
    """
    Simulates realistic used car data based on European market patterns
    similar to what AUTO1 Group would encounter
    """
    
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        self.makes = ['BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Ford', 'Peugeot', 
                     'Renault', 'Opel', 'Fiat', 'Toyota', 'Honda', 'Nissan']
        self.fuel_types = ['Petrol', 'Diesel', 'Electric', 'Hybrid']
        self.body_types = ['Sedan', 'Hatchback', 'SUV', 'Estate', 'Coupe', 'Convertible']
        self.locations = ['Berlin', 'Munich', 'Hamburg', 'Frankfurt', 'Cologne', 
                         'Stuttgart', 'Paris', 'London', 'Madrid', 'Rome']
        
    def generate_realistic_data(self):
        """
        Generate synthetic but realistic car data with correlated features
        """
        np.random.seed(42)
        data = []
        
        for i in range(self.n_samples):
            # Base car characteristics
            make = np.random.choice(self.makes, p=[0.12, 0.11, 0.10, 0.15, 0.08, 0.07, 
                                                  0.07, 0.06, 0.05, 0.08, 0.06, 0.05])
            
            # Year influences many other factors
            year = np.random.randint(2010, 2024)
            age = 2024 - year
            
            # Mileage correlated with age
            base_mileage = age * np.random.uniform(8000, 25000)
            mileage = max(1000, int(base_mileage + np.random.normal(0, 10000)))
            
            fuel_type = np.random.choice(self.fuel_types, 
                                       p=[0.35, 0.45, 0.05, 0.15] if year < 2018 
                                         else [0.25, 0.35, 0.15, 0.25])
            
            body_type = np.random.choice(self.body_types)
            location = np.random.choice(self.locations)
            
            # Engine size depends on make and fuel type
            if make in ['BMW', 'Mercedes', 'Audi']:
                engine_size = np.random.uniform(1.8, 4.0)
            else:
                engine_size = np.random.uniform(1.0, 2.5)
                
            if fuel_type == 'Electric':
                engine_size = 0.0
                
            # Market demand simulation (seasonal and location-based)
            base_demand = np.random.uniform(0.3, 0.9)
            if location in ['Berlin', 'Munich', 'London', 'Paris']:
                base_demand *= 1.2  # Higher demand in major cities
            if fuel_type == 'Electric':
                base_demand *= 1.1  # Growing EV demand
                
            market_demand = min(1.0, base_demand)
            
            # Condition score (1-10)
            condition = max(1, min(10, int(np.random.normal(7, 1.5))))
            if age > 10:
                condition = max(1, condition - np.random.randint(0, 3))
                
            # Days in inventory (AUTO1's key metric)
            days_in_inventory = max(1, int(np.random.exponential(20)))
            
            # Previous owners
            prev_owners = min(5, max(1, int(np.random.poisson(1.5) + 1)))
            if age > 8:
                prev_owners += np.random.randint(0, 2)
                
            # Calculate realistic price based on multiple factors
            base_price = self._calculate_base_price(make, year, fuel_type, body_type)
            
            # Apply depreciation and adjustments
            price = self._apply_price_adjustments(base_price, age, mileage, condition, 
                                                market_demand, fuel_type, location)
            
            data.append({
                'make': make,
                'model': f"{make}_Model_{np.random.randint(1, 6)}",
                'year': year,
                'mileage': mileage,
                'fuel_type': fuel_type,
                'body_type': body_type,
                'engine_size': round(engine_size, 1),
                'location': location,
                'condition_score': condition,
                'previous_owners': prev_owners,
                'market_demand': round(market_demand, 3),
                'days_in_inventory': days_in_inventory,
                'selling_price': int(price)
            })
            
        return pd.DataFrame(data)
    
    def _calculate_base_price(self, make, year, fuel_type, body_type):
        """Calculate base price before adjustments"""
        # Premium brands command higher prices
        brand_multipliers = {
            'BMW': 1.3, 'Mercedes': 1.35, 'Audi': 1.25,
            'Volkswagen': 1.0, 'Toyota': 1.1, 'Honda': 1.05,
            'Ford': 0.85, 'Peugeot': 0.8, 'Renault': 0.8,
            'Opel': 0.75, 'Fiat': 0.7, 'Nissan': 0.9
        }
        
        # Body type affects pricing
        body_multipliers = {
            'SUV': 1.2, 'Coupe': 1.15, 'Convertible': 1.1,
            'Estate': 1.05, 'Sedan': 1.0, 'Hatchback': 0.95
        }
        
        # Fuel type premium
        fuel_multipliers = {
            'Electric': 1.3, 'Hybrid': 1.15, 'Diesel': 1.05, 'Petrol': 1.0
        }
        
        # Base calculation
        age = 2024 - year
        base = 25000 * brand_multipliers.get(make, 1.0)
        base *= body_multipliers.get(body_type, 1.0)
        base *= fuel_multipliers.get(fuel_type, 1.0)
        
        return base
    
    def _apply_price_adjustments(self, base_price, age, mileage, condition, 
                               market_demand, fuel_type, location):
        """Apply various price adjustments"""
        price = base_price
        
        # Age depreciation (non-linear)
        depreciation_rate = 0.12 if fuel_type != 'Electric' else 0.08
        price *= (1 - depreciation_rate) ** age
        
        # Mileage adjustment
        avg_mileage_per_year = 15000
        expected_mileage = age * avg_mileage_per_year
        mileage_diff = (mileage - expected_mileage) / 10000
        price *= (1 - mileage_diff * 0.05)
        
        # Condition adjustment
        condition_multiplier = 0.7 + (condition / 10) * 0.6
        price *= condition_multiplier
        
        # Market demand effect
        price *= (0.8 + market_demand * 0.4)
        
        # Location premium/discount
        location_multipliers = {
            'Berlin': 1.05, 'Munich': 1.08, 'London': 1.15, 'Paris': 1.1,
            'Hamburg': 1.02, 'Frankfurt': 1.06, 'Stuttgart': 1.04,
            'Cologne': 1.0, 'Madrid': 0.95, 'Rome': 0.92
        }
        price *= location_multipliers.get(location, 1.0)
        
        # Add some random noise
        price *= np.random.uniform(0.9, 1.1)
        
        return max(1000, price)  # Minimum price floor

# Generate the dataset
print("Generating realistic used car dataset...")
data_gen = DataGenerator(n_samples=8000)
df = data_gen.generate_realistic_data()

print(f"Dataset generated with {len(df)} records")
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# ================================================================
# 3. DATA CLEANING
# ================================================================

class DataCleaner:
    """
    Comprehensive data cleaning for automotive pricing models
    """
    
    def __init__(self):
        self.numeric_features = ['year', 'mileage', 'engine_size', 'condition_score', 
                               'previous_owners', 'market_demand', 'days_in_inventory']
        self.categorical_features = ['make', 'fuel_type', 'body_type', 'location']
        
    def clean_data(self, df):
        """
        Complete data cleaning pipeline
        """
        df_clean = df.copy()
        
        print("Starting data cleaning process...")
        
        # 1. Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # 2. Detect and handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # 3. Data type corrections
        df_clean = self._correct_data_types(df_clean)
        
        # 4. Create validation flags
        df_clean = self._create_validation_flags(df_clean)
        
        print(f"Data cleaning completed. Rows: {len(df)} -> {len(df_clean)}")
        
        return df_clean
    
    def _handle_missing_values(self, df):
        """
        Handle missing values with domain-specific logic
        """
        print("Handling missing values...")
        
        # For automotive data, certain missing patterns are meaningful
        for col in self.numeric_features:
            if col in df.columns:
                if col == 'engine_size':
                    # Electric cars have 0 engine size
                    df[col] = df[col].fillna(df.groupby('fuel_type')[col].transform('median'))
                elif col == 'condition_score':
                    # Use age-based imputation for condition
                    df[col] = df[col].fillna(df.groupby(pd.cut(df['year'], bins=5))[col].transform('median'))
                else:
                    # General median imputation
                    df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical features
        for col in self.categorical_features:
            if col in df.columns:
                # Use mode for categorical
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _handle_outliers(self, df):
        """
        Detect and handle outliers using automotive domain knowledge
        """
        print("Detecting and handling outliers...")
        
        # Price outliers (beyond reasonable range)
        q1_price = df['selling_price'].quantile(0.01)
        q99_price = df['selling_price'].quantile(0.99)
        
        print(f"Price outliers: < {q1_price:.0f} or > {q99_price:.0f}")
        
        # Mileage outliers (unrealistic mileage for age)
        df['age'] = 2024 - df['year']
        df['mileage_per_year'] = df['mileage'] / df['age'].clip(lower=1)
        
        # Remove cars with impossible mileage (>50k km/year consistently)
        outlier_mask = (
            (df['selling_price'] < q1_price) | (df['selling_price'] > q99_price) |
            (df['mileage_per_year'] > 50000) |
            (df['mileage'] < 100)  # Suspiciously low mileage
        )
        
        print(f"Removing {outlier_mask.sum()} outliers ({outlier_mask.mean():.2%} of data)")
        
        df_clean = df[~outlier_mask].copy()
        df_clean.drop(['mileage_per_year'], axis=1, inplace=True)
        
        return df_clean
    
    def _correct_data_types(self, df):
        """
        Ensure correct data types for modeling
        """
        print("Correcting data types...")
        
        # Convert to appropriate types
        df['year'] = df['year'].astype(int)
        df['mileage'] = df['mileage'].astype(int)
        df['condition_score'] = df['condition_score'].astype(int)
        df['previous_owners'] = df['previous_owners'].astype(int)
        df['days_in_inventory'] = df['days_in_inventory'].astype(int)
        df['selling_price'] = df['selling_price'].astype(int)
        
        # Categorical as category type for memory efficiency
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df
    
    def _create_validation_flags(self, df):
        """
        Create flags for data quality validation
        """
        # Create flags for potential data quality issues
        df['is_low_mileage'] = df['mileage'] < (df['age'] * 5000)  # Very low usage
        df['is_high_mileage'] = df['mileage'] > (df['age'] * 30000)  # Very high usage
        df['is_premium_brand'] = df['make'].isin(['BMW', 'Mercedes', 'Audi'])
        
        return df

# Apply data cleaning
cleaner = DataCleaner()
df_clean = cleaner.clean_data(df)

print("\nCleaned dataset summary:")
print(df_clean.describe())

# ================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================

class EDAnalyzer:
    """
    Comprehensive EDA for automotive pricing analysis
    """
    
    def __init__(self, df):
        self.df = df
        
    def perform_eda(self):
        """
        Complete EDA pipeline
        """
        print("Starting Exploratory Data Analysis...")
        
        self._basic_statistics()
        self._price_distribution_analysis()
        self._feature_correlation_analysis()
        self._categorical_analysis()
        self._business_insights()
    
    def _basic_statistics(self):
        """
        Basic statistical overview
        """
        print("\n" + "="*50)
        print("BASIC STATISTICS")
        print("="*50)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Average selling price: €{self.df['selling_price'].mean():,.0f}")
        print(f"Median selling price: €{self.df['selling_price'].median():,.0f}")
        print(f"Price range: €{self.df['selling_price'].min():,.0f} - €{self.df['selling_price'].max():,.0f}")
        print(f"Average age: {self.df['age'].mean():.1f} years")
        print(f"Average mileage: {self.df['mileage'].mean():,.0f} km")
        
    def _price_distribution_analysis(self):
        """
        Analyze price distributions
        """
        print("\n" + "="*50)
        print("PRICE DISTRIBUTION ANALYSIS")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Price distribution
        axes[0,0].hist(self.df['selling_price'], bins=50, alpha=0.7, color='blue')
        axes[0,0].set_title('Price Distribution')
        axes[0,0].set_xlabel('Selling Price (€)')
        axes[0,0].set_ylabel('Frequency')
        
        # Log price distribution (often more normal)
        axes[0,1].hist(np.log(self.df['selling_price']), bins=50, alpha=0.7, color='green')
        axes[0,1].set_title('Log Price Distribution')
        axes[0,1].set_xlabel('Log(Selling Price)')
        axes[0,1].set_ylabel('Frequency')
        
        # Price by make
        price_by_make = self.df.groupby('make')['selling_price'].median().sort_values(ascending=False)
        axes[1,0].bar(range(len(price_by_make)), price_by_make.values)
        axes[1,0].set_title('Median Price by Make')
        axes[1,0].set_xlabel('Car Make')
        axes[1,0].set_ylabel('Median Price (€)')
        axes[1,0].set_xticks(range(len(price_by_make)))
        axes[1,0].set_xticklabels(price_by_make.index, rotation=45)
        
        # Price vs Age
        axes[1,1].scatter(self.df['age'], self.df['selling_price'], alpha=0.5)
        axes[1,1].set_title('Price vs Age')
        axes[1,1].set_xlabel('Age (years)')
        axes[1,1].set_ylabel('Selling Price (€)')
        
        plt.tight_layout()
        plt.show()
        
        # Price statistics by key segments
        print("\nPrice statistics by premium vs non-premium brands:")
        premium_analysis = self.df.groupby('is_premium_brand')['selling_price'].agg(['mean', 'median', 'std'])
        print(premium_analysis)
    
    def _feature_correlation_analysis(self):
        """
        Analyze correlations between features and price
        """
        print("\n" + "="*50)
        print("FEATURE CORRELATION ANALYSIS")
        print("="*50)
        
        # Select numeric features for correlation
        numeric_features = ['selling_price', 'year', 'age', 'mileage', 'engine_size', 
                          'condition_score', 'previous_owners', 'market_demand', 'days_in_inventory']
        
        corr_matrix = self.df[numeric_features].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Price correlations
        price_corr = corr_matrix['selling_price'].sort_values(key=abs, ascending=False)
        print("\nCorrelation with selling price:")
        for feature, corr in price_corr.items():
            if feature != 'selling_price':
                print(f"{feature:20s}: {corr:6.3f}")
    
    def _categorical_analysis(self):
        """
        Analyze categorical features impact on price
        """
        print("\n" + "="*50)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Price by fuel type
        fuel_prices = self.df.groupby('fuel_type')['selling_price'].median().sort_values(ascending=False)
        axes[0,0].bar(fuel_prices.index, fuel_prices.values, color='skyblue')
        axes[0,0].set_title('Median Price by Fuel Type')
        axes[0,0].set_ylabel('Median Price (€)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Price by body type
        body_prices = self.df.groupby('body_type')['selling_price'].median().sort_values(ascending=False)
        axes[0,1].bar(body_prices.index, body_prices.values, color='lightcoral')
        axes[0,1].set_title('Median Price by Body Type')
        axes[0,1].set_ylabel('Median Price (€)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Price by location
        location_prices = self.df.groupby('location')['selling_price'].median().sort_values(ascending=False)
        axes[1,0].bar(location_prices.index, location_prices.values, color='lightgreen')
        axes[1,0].set_title('Median Price by Location')
        axes[1,0].set_ylabel('Median Price (€)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Price distribution by condition score
        self.df.boxplot(column='selling_price', by='condition_score', ax=axes[1,1])
        axes[1,1].set_title('Price Distribution by Condition Score')
        axes[1,1].set_xlabel('Condition Score')
        axes[1,1].set_ylabel('Selling Price (€)')
        
        plt.tight_layout()
        plt.show()
    
    def _business_insights(self):
        """
        Generate business-relevant insights
        """
        print("\n" + "="*50)
        print("BUSINESS INSIGHTS FOR AUTO1-TYPE OPERATIONS")
        print("="*50)
        
        # Inventory turnover analysis
        avg_days_inventory = self.df['days_in_inventory'].mean()
        print(f"Average days in inventory: {avg_days_inventory:.1f} days")
        
        slow_moving = self.df[self.df['days_in_inventory'] > 30]
        print(f"Cars with >30 days inventory: {len(slow_moving)} ({len(slow_moving)/len(self.df):.1%})")
        
        # High-value vs low-value segments
        high_value = self.df[self.df['selling_price'] > self.df['selling_price'].quantile(0.75)]
        low_value = self.df[self.df['selling_price'] <= self.df['selling_price'].quantile(0.25)]
        
        print(f"\nHigh-value segment analysis (top 25% by price):")
        print(f"Average price: €{high_value['selling_price'].mean():,.0f}")
        print(f"Average days in inventory: {high_value['days_in_inventory'].mean():.1f}")
        print(f"Most common make: {high_value['make'].mode()[0]}")
        
        print(f"\nLow-value segment analysis (bottom 25% by price):")
        print(f"Average price: €{low_value['selling_price'].mean():,.0f}")
        print(f"Average days in inventory: {low_value['days_in_inventory'].mean():.1f}")
        print(f"Most common make: {low_value['make'].mode()[0]}")
        
        # Market demand impact
        high_demand = self.df[self.df['market_demand'] > 0.7]
        low_demand = self.df[self.df['market_demand'] < 0.4]
        
        print(f"\nMarket demand impact:")
        print(f"High demand cars (>0.7): avg price €{high_demand['selling_price'].mean():,.0f}, avg inventory {high_demand['days_in_inventory'].mean():.1f} days")
        print(f"Low demand cars (<0.4): avg price €{low_demand['selling_price'].mean():,.0f}, avg inventory {low_demand['days_in_inventory'].mean():.1f} days")

# Perform EDA
eda = EDAnalyzer(df_clean)
eda.perform_eda()

# ================================================================
# 5. FEATURE ENGINEERING
# ================================================================

class FeatureEngineer:
    """
    Create advanced features for automotive pricing models
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def engineer_features(self, df):
        """
        Complete feature engineering pipeline
        """
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        df_eng = df.copy()
        
        # 1. Create derived features
        df_eng = self._create_derived_features(df_eng)
        
        # 2. Create interaction features
        df_eng = self._create_interaction_features(df_eng)
        
        # 3. Create categorical features
        df_eng = self._create_categorical_features(df_eng)
        
        # 4. Create business-specific features
        df_eng = self._create_business_features(df_eng)
        
        print(f"Feature engineering completed. Features: {len(df.columns)} -> {len(df_eng.columns)}")
        
        return df_eng
    
    def _create_derived_features(self, df):
        """
        Create basic derived features
        """
        print("Creating derived features...")
        
        # Age-related features
        if 'age' not in df.columns:
            df['age'] = 2024 - df['year']
        
        # Mileage-related features
        df['mileage_per_year'] = df['mileage'] / df['age'].clip(lower=1)
        df['mileage_category'] = pd.cut(df['mileage'], 
                                      bins=[0, 50000, 100000, 150000, float('inf')],
                                      labels=['Low', 'Medium', 'High', 'Very_High'])
        
        # Price per kilometer (efficiency metric)
        df['price_per_km'] = df['selling_price'] / df['mileage'].clip(lower=1)
        
        # Age groups for different depreciation patterns
        df['age_group'] = pd.cut(df['age'], 
                               bins=[0, 3, 7, 12, float('inf')],
                               labels=['New', 'Recent', 'Mature', 'Old'])
        
        # Condition-to-age ratio (how well maintained)
        df['condition_age_ratio'] = df['condition_score'] / df['age'].clip(lower=1)
        
        return df
    
    def _create_interaction_features(self, df):
        """
        Create interaction features that are important for pricing
        """
        print("Creating interaction features...")
        
        # Brand-Age interaction (luxury brands depreciate differently)
        df['brand_age_interaction'] = df['is_premium_brand'].astype(int) * df['age']
        
        # Mileage-Condition interaction
        df['mileage_condition_interaction'] = df['mileage'] * df['condition_score']
        
        # Market demand and days in inventory (pricing pressure)
        df['demand_inventory_pressure'] = df['market_demand'] / (df['days_in_inventory'] + 1)
        
        # Engine size efficiency (important for fuel economy)
        df['power_efficiency'] = df['engine_size'] / df['age'].clip(lower=1)
        
        return df
    
    def _create_categorical_features(self, df):
        """
        Engineer categorical features for better model performance
        """
        print("Engineering categorical features...")
        
        # Fuel type evolution (EV premium changes over time)
        df['fuel_age_segment'] = df['fuel_type'] + '_' + df['age_group'].astype(str)
        
        # Location-Brand combination (some brands are more popular in certain regions)
        df['location_brand'] = df['location'] + '_' + df['make']
        
        # Body type and fuel type combination
        df['body_fuel_combo'] = df['body_type'] + '_' + df['fuel_type']
        
        return df
    
    def _create_business_features(self, df):
        """
        Create features specific to AUTO1's business model
        """
        print("Creating business-specific features...")
        
        # Inventory velocity indicator
        df['inventory_velocity'] = 1 / (df['days_in_inventory'] + 1)
        
        # Profit potential (higher condition, lower age = higher margins)
        df['profit_potential'] = (df['condition_score'] * 10) / (df['age'] + 1)
        
        # Market position (price relative to similar cars)
        price_segments = df.groupby(['make', 'age_group'])['selling_price'].transform('median')
        df['price_position'] = df['selling_price'] / price_segments
        
        # Ownership stability (fewer owners usually better)
        df['ownership_stability'] = 1 / df['previous_owners']
        
        # Total cost of ownership proxy
        df['tco_proxy'] = df['selling_price'] + (df['age'] * 500) + (df['mileage'] * 0.1)
        
        # Risk score (combination of factors that affect saleability)
        df['risk_score'] = (
            (df['age'] * 0.1) + 
            (df['mileage'] / 100000) + 
            ((10 - df['condition_score']) * 0.2) +
            (df['previous_owners'] * 0.1) +
            ((1 - df['market_demand']) * 0.3)
        )
        
        return df

# Apply feature engineering
engineer = FeatureEngineer()
df_features = engineer.engineer_features(df_clean)

print(f"\nEngineered features overview:")
print(f"Total features: {len(df_features.columns)}")
print(f"New features created: {len(df_features.columns) - len(df_clean.columns)}")

# ================================================================
# 6. MODEL SELECTION & TRAINING
# ================================================================

class ModelTrainer:
    """
    Train and compare multiple models for automotive pricing
    """
    
    def __init__(self):
        self.models = {}
        self.performance = {}
        self.best_model = None
        self.feature_cols = None
        self.target_col = 'selling_price'
        
    def prepare_data(self, df):
        """
        Prepare data for modeling with proper encoding
        """
        print("\n" + "="*50)
        print("DATA PREPARATION FOR MODELING")
        print("="*50)
        
        df_model = df.copy()
        
        # Select features for modeling
        # Exclude target and identifier columns
        exclude_cols = ['selling_price', 'model', 'is_low_mileage', 'is_high_mileage']
        
        # Get categorical columns that need encoding
        categorical_cols = df_model.select_dtypes(include=['category', 'object']).columns
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
        
        # Remove target and unwanted columns
        feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
        
        X = df_encoded[feature_cols]
        y = df_encoded[self.target_col]
        
        print(f"Features for modeling: {len(feature_cols)}")
        print(f"Training samples: {len(X)}")
        
        self.feature_cols = feature_cols
        
        return X, y
    
    def train_models(self, X, y):
        """
        Train multiple models and compare performance
        """
        print("\n" + "="*50)
        print("MODEL TRAINING & COMPARISON")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_root_mean_squared_error')
            cv_rmse = -cv_scores.mean()
            
            results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_rmse': cv_rmse,
                'predictions_test': y_pred_test
            }
            
            print(f"  Test RMSE: €{test_rmse:,.0f}")
            print(f"  Test MAE:  €{test_mae:,.0f}")
            print(f"  Test R²:   {test_r2:.3f}")
            print(f"  CV RMSE:   €{cv_rmse:,.0f}")
        
        # Select best model based on test RMSE
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Test RMSE: €{results[best_model_name]['test_rmse']:,.0f}")
        print(f"   Test R²: {results[best_model_name]['test_r2']:.3f}")
        
        # Store results
        self.models = {name: result['model'] for name, result in results.items()}
        self.performance = results
        
        # Create comparison visualization
        self._plot_model_comparison(results)
        
        return X_train, X_test, y_train, y_test
    
    def _plot_model_comparison(self, results):
        """
        Visualize model performance comparison
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        models = list(results.keys())
        
        # RMSE comparison
        test_rmse = [results[model]['test_rmse'] for model in models]
        cv_rmse = [results[model]['cv_rmse'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x - width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
        axes[0].bar(x + width/2, cv_rmse, width, label='CV RMSE', alpha=0.8)
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('RMSE (€)')
        axes[0].set_title('Model RMSE Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # R² comparison
        test_r2 = [results[model]['test_r2'] for model in models]
        axes[1].bar(models, test_r2, alpha=0.8, color='green')
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('Model R² Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Prediction scatter for best model
        best_model_name = min(models, key=lambda x: results[x]['test_rmse'])
        y_test_actual = list(results.values())[0]  # We need actual y_test values
        # This is a simplified version - in practice you'd store y_test
        axes[2].set_title(f'Best Model ({best_model_name}) - Prediction Quality')
        axes[2].set_xlabel('Actual Price (€)')
        axes[2].set_ylabel('Predicted Price (€)')
        axes[2].text(0.05, 0.95, f'R² = {results[best_model_name]["test_r2"]:.3f}', 
                    transform=axes[2].transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        plt.show()

# Train models
trainer = ModelTrainer()
X, y = trainer.prepare_data(df_features)
X_train, X_test, y_train, y_test = trainer.train_models(X, y)

# ================================================================
# 7. DYNAMIC PRICING LOGIC
# ================================================================

class DynamicPricingEngine:
    """
    Implement dynamic pricing logic for AUTO1-style operations
    """
    
    def __init__(self, base_model, feature_cols):
        self.base_model = base_model
        self.feature_cols = feature_cols
        self.pricing_rules = self._initialize_pricing_rules()
        
    def _initialize_pricing_rules(self):
        """
        Initialize business rules for dynamic pricing
        """
        return {
            'inventory_pressure': {
                'high_inventory_days': 30,  # Days after which to apply pressure
                'discount_rate': 0.02,      # 2% discount per week
                'max_discount': 0.15        # Maximum 15% discount
            },
            'market_demand': {
                'high_demand_threshold': 0.8,
                'low_demand_threshold': 0.4,
                'demand_multiplier': 0.1    # ±10% based on demand
            },
            'seasonal_factors': {
                'convertible_summer_boost': 1.1,
                'suv_winter_boost': 1.05,
                'electric_urban_boost': 1.08
            },
            'competition': {
                'undercut_threshold': 0.05,  # Undercut by 5% if needed
                'premium_threshold': 1.1     # Premium positioning limit
            }
        }
    
    def calculate_dynamic_price(self, car_features, market_conditions=None):
        """
        Calculate dynamic price based on car features and market conditions
        """
        # Get base ML prediction
        base_price = self.base_model.predict([car_features])[0]
        
        # Apply dynamic adjustments
        adjusted_price = base_price
        adjustments = {}
        
        # 1. Inventory pressure adjustment
        days_in_inventory = car_features[self.feature_cols.index('days_in_inventory')] if 'days_in_inventory' in self.feature_cols else 15
        inventory_adjustment = self._calculate_inventory_adjustment(days_in_inventory)
        adjusted_price *= inventory_adjustment
        adjustments['inventory'] = inventory_adjustment
        
        # 2. Market demand adjustment
        market_demand = car_features[self.feature_cols.index('market_demand')] if 'market_demand' in self.feature_cols else 0.5
        demand_adjustment = self._calculate_demand_adjustment(market_demand)
        adjusted_price *= demand_adjustment
        adjustments['demand'] = demand_adjustment
        
        # 3. Seasonal adjustment (simplified)
        seasonal_adjustment = self._calculate_seasonal_adjustment(car_features)
        adjusted_price *= seasonal_adjustment
        adjustments['seasonal'] = seasonal_adjustment
        
        # 4. Competition adjustment (if market data available)
        if market_conditions and 'competitor_prices' in market_conditions:
            competition_adjustment = self._calculate_competition_adjustment(
                adjusted_price, market_conditions['competitor_prices']
            )
            adjusted_price *= competition_adjustment
            adjustments['competition'] = competition_adjustment
        
        return {
            'base_price': base_price,
            'final_price': max(1000, adjusted_price),  # Price floor
            'adjustments': adjustments,
            'total_adjustment': adjusted_price / base_price
        }
    
    def _calculate_inventory_adjustment(self, days_in_inventory):
        """
        Adjust price based on inventory age
        """
        rules = self.pricing_rules['inventory_pressure']
        
        if days_in_inventory <= rules['high_inventory_days']:
            return 1.0  # No adjustment
        
        # Calculate discount based on excess days
        excess_days = days_in_inventory - rules['high_inventory_days']
        weeks_excess = excess_days / 7
        discount = min(weeks_excess * rules['discount_rate'], rules['max_discount'])
        
        return 1 - discount
    
    def _calculate_demand_adjustment(self, market_demand):
        """
        Adjust price based on market demand
        """
        rules = self.pricing_rules['market_demand']
        
        if market_demand > rules['high_demand_threshold']:
            # High demand - increase price
            premium = (market_demand - rules['high_demand_threshold']) * rules['demand_multiplier'] * 2
            return 1 + premium
        elif market_demand < rules['low_demand_threshold']:
            # Low demand - decrease price
            discount = (rules['low_demand_threshold'] - market_demand) * rules['demand_multiplier'] * 2
            return 1 - discount
        else:
            # Normal demand
            return 1.0
    
    def _calculate_seasonal_adjustment(self, car_features):
        """
        Apply seasonal adjustments (simplified version)
        """
        # This would typically use current month/season
        # For demo, we'll use car characteristics
        
        adjustment = 1.0
        
        # Check if we can identify body type from features
        # In practice, you'd have more sophisticated seasonal logic
        # This is a simplified demonstration
        
        return adjustment
    
    def _calculate_competition_adjustment(self, current_price, competitor_prices):
        """
        Adjust price based on competition
        """
        if not competitor_prices:
            return 1.0
        
        avg_competitor_price = np.mean(competitor_prices)
        
        rules = self.pricing_rules['competition']
        
        if current_price > avg_competitor_price * rules['premium_threshold']:
            # We're pricing too high compared to competition
            return 0.95  # Reduce by 5%
        elif current_price < avg_competitor_price * (1 - rules['undercut_threshold']):
            # We're pricing too low - might be leaving money on table
            return 1.02  # Increase by 2%
        
        return 1.0
    
    def batch_pricing_update(self, inventory_df, market_conditions=None):
        """
        Update prices for entire inventory
        """
        print("\n" + "="*50)
        print("BATCH DYNAMIC PRICING UPDATE")
        print("="*50)
        
        pricing_results = []
        
        for idx, row in inventory_df.iterrows():
            # Prepare features for this car
            car_features = [row[col] if col in row else 0 for col in self.feature_cols]
            
            # Calculate dynamic price
            pricing_result = self.calculate_dynamic_price(car_features, market_conditions)
            
            pricing_results.append({
                'car_id': idx,
                'make': row.get('make', 'Unknown'),
                'model': row.get('model', 'Unknown'),
                'current_price': row.get('selling_price', 0),
                'base_ml_price': pricing_result['base_price'],
                'recommended_price': pricing_result['final_price'],
                'price_change': pricing_result['final_price'] - row.get('selling_price', 0),
                'price_change_pct': (pricing_result['final_price'] - row.get('selling_price', 0)) / row.get('selling_price', 1),
                'adjustments': pricing_result['adjustments'],
                'days_in_inventory': row.get('days_in_inventory', 0)
            })
        
        results_df = pd.DataFrame(pricing_results)
        
        # Summary statistics
        print(f"Processed {len(results_df)} vehicles")
        print(f"Average price change: €{results_df['price_change'].mean():,.0f}")
        print(f"Average price change %: {results_df['price_change_pct'].mean():.1%}")
        print(f"Vehicles requiring price increase: {(results_df['price_change'] > 0).sum()}")
        print(f"Vehicles requiring price decrease: {(results_df['price_change'] < 0).sum()}")
        
        return results_df

# Initialize dynamic pricing engine
pricing_engine = DynamicPricingEngine(trainer.best_model, trainer.feature_cols)

# Demonstrate batch pricing on a sample
sample_inventory = df_features.sample(100, random_state=42)
pricing_results = pricing_engine.batch_pricing_update(sample_inventory)

print("\nSample of pricing recommendations:")
print(pricing_results[['make', 'current_price', 'recommended_price', 'price_change_pct', 'days_in_inventory']].head(10))

# ================================================================
# 8. PERFORMANCE EVALUATION
# ================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation with business metrics
    """
    
    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        
    def evaluate_models(self):
        """
        Comprehensive model evaluation
        """
        print("\n" + "="*50)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*50)
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Basic predictions
            y_pred = model.predict(self.X_test)
            
            # Statistical metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
            
            # Business metrics
            business_metrics = self._calculate_business_metrics(self.y_test, y_pred)
            
            evaluation_results[name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'predictions': y_pred,
                **business_metrics
            }
            
            print(f"  RMSE: €{rmse:,.0f}")
            print(f"  MAE:  €{mae:,.0f}")
            print(f"  R²:   {r2:.3f}")
            print(f"  MAPE: {mape:.1f}%")
            print(f"  Profit Impact: €{business_metrics['profit_impact']:,.0f}")
            print(f"  Pricing Accuracy: {business_metrics['pricing_accuracy']:.1%}")
        
        # Create evaluation visualization
        self._create_evaluation_plots(evaluation_results)
        
        return evaluation_results
    
    def _calculate_business_metrics(self, y_true, y_pred):
        """
        Calculate business-relevant metrics for AUTO1-type operations
        """
        # Pricing accuracy (% of predictions within acceptable range)
        acceptable_error = 0.10  # 10% error acceptable
        accurate_predictions = np.abs((y_true - y_pred) / y_true) <= acceptable_error
        pricing_accuracy = accurate_predictions.mean()
        
        # Revenue impact (sum of prediction errors)
        revenue_impact = np.sum(y_pred - y_true)
        
        # Profit impact (assuming 10% margin)
        margin_rate = 0.10
        profit_impact = revenue_impact * margin_rate
        
        # Risk metrics (large underpricing is risky)
        underpricing_risk = np.sum(np.maximum(0, y_true - y_pred))
        overpricing_risk = np.sum(np.maximum(0, y_pred - y_true))
        
        # Inventory velocity impact (faster sales with better pricing)
        # Simplified: better predictions lead to faster sales
        avg_error = np.mean(np.abs(y_true - y_pred))
        velocity_impact = max(0, 5000 - avg_error) / 1000  # Simplified metric
        
        return {
            'pricing_accuracy': pricing_accuracy,
            'revenue_impact': revenue_impact,
            'profit_impact': profit_impact,
            'underpricing_risk': underpricing_risk,
            'overpricing_risk': overpricing_risk,
            'velocity_impact': velocity_impact
        }
    
    def _create_evaluation_plots(self, results):
        """
        Create comprehensive evaluation visualizations
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        models = list(results.keys())
        colors = ['blue', 'green', 'red', 'orange']
        
        # 1. RMSE Comparison
        rmse_values = [results[model]['rmse'] for model in models]
        axes[0,0].bar(models, rmse_values, color=colors[:len(models)], alpha=0.7)
        axes[0,0].set_title('RMSE Comparison')
        axes[0,0].set_ylabel('RMSE (€)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. R² Comparison
        r2_values = [results[model]['r2'] for model in models]
        axes[0,1].bar(models, r2_values, color=colors[:len(models)], alpha=0.7)
        axes[0,1].set_title('R² Score Comparison')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Business Impact Comparison
        profit_impact = [results[model]['profit_impact'] for model in models]
        axes[0,2].bar(models, profit_impact, color=colors[:len(models)], alpha=0.7)
        axes[0,2].set_title('Profit Impact Comparison')
        axes[0,2].set_ylabel('Profit Impact (€)')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Prediction vs Actual (Best model)
        best_model = min(models, key=lambda x: results[x]['rmse'])
        best_predictions = results[best_model]['predictions']
        
        axes[1,0].scatter(self.y_test, best_predictions, alpha=0.6, color='blue')
        axes[1,0].plot([self.y_test.min(), self.y_test.max()], 
                      [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1,0].set_xlabel('Actual Price (€)')
        axes[1,0].set_ylabel('Predicted Price (€)')
        axes[1,0].set_title(f'Prediction Quality - {best_model}')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Residuals Analysis
        residuals = self.y_test - best_predictions
        axes[1,1].scatter(best_predictions, residuals, alpha=0.6, color='green')
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('Predicted Price (€)')
        axes[1,1].set_ylabel('Residuals (€)')
        axes[1,1].set_title('Residuals Analysis')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Pricing Accuracy
        accuracy_values = [results[model]['pricing_accuracy'] for model in models]
        axes[1,2].bar(models, accuracy_values, color=colors[:len(models)], alpha=0.7)
        axes[1,2].set_title('Pricing Accuracy (±10%)')
        axes[1,2].set_ylabel('Accuracy Rate')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Evaluate models
evaluator = ModelEvaluator(trainer.models, X_test, y_test)
evaluation_results = evaluator.evaluate_models()

# ================================================================
# 9. MODEL INTERPRETABILITY
# ================================================================

class ModelInterpreter:
    """
    Provide model interpretability for pricing transparency
    """
    
    def __init__(self, model, feature_names, X_test, y_test):
        self.model = model
        self.feature_names = feature_names
        self.X_test = X_test
        self.y_test = y_test
        
    def analyze_feature_importance(self):
        """
        Analyze and visualize feature importance
        """
        print("\n" + "="*50)
        print("MODEL INTERPRETABILITY ANALYSIS")
        print("="*50)
        
        # Tree-based models have built-in feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 15 Most Important Features:")
            print(feature_importance_df.head(15))
            
            # Visualize top features
            self._plot_feature_importance(feature_importance_df.head(20))
            
            return feature_importance_df
        else:
            print("Model doesn't support feature importance analysis")
            return None
    
    def _plot_feature_importance(self, importance_df):
        """
        Plot feature importance
        """
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar plot
        plt.barh(range(len(importance_df)), importance_df['importance'], alpha=0.8)
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances for Price Prediction')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_business_insights(self, importance_df):
        """
        Generate business insights from feature importance
        """
        if importance_df is None:
            return
            
        print("\n" + "="*50)
        print("BUSINESS INSIGHTS FROM MODEL")
        print("="*50)
        
        top_features = importance_df.head(10)
        
        insights = []
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if 'age' in feature.lower():
                insights.append(f"🔹 Car age is critical (importance: {importance:.3f}) - Depreciation heavily impacts pricing")
            elif 'mileage' in feature.lower():
                insights.append(f"🔹 Mileage significantly affects price (importance: {importance:.3f}) - Usage history is key")
            elif 'condition' in feature.lower():
                insights.append(f"🔹 Condition score is important (importance: {importance:.3f}) - Quality assessment drives value")
            elif 'premium' in feature.lower() or any(brand in feature for brand in ['BMW', 'Mercedes', 'Audi']):
                insights.append(f"🔹 Brand premium matters (importance: {importance:.3f}) - Luxury brands hold value better")
            elif 'demand' in feature.lower():
                insights.append(f"🔹 Market demand affects pricing (importance: {importance:.3f}) - Supply/demand dynamics crucial")
            elif 'fuel' in feature.lower():
                insights.append(f"🔹 Fuel type influences price (importance: {importance:.3f}) - EV/Hybrid premium visible")
            elif 'location' in feature.lower():
                insights.append(f"🔹 Location impacts pricing (importance: {importance:.3f}) - Geographic price variations exist")
        
        print("Key Business Insights:")
        for insight in insights:
            print(insight)
        
        print(f"\n💡 Model Recommendations for AUTO1:")
        print(f"   • Focus on accurate age and mileage data collection")
        print(f"   • Invest in better condition assessment tools")
        print(f"   • Monitor regional market demand trends")
        print(f"   • Consider brand-specific pricing strategies")
        print(f"   • Track fuel type market evolution (EV growth)")

# Analyze model interpretability
if hasattr(trainer.best_model, 'feature_importances_'):
    interpreter = ModelInterpreter(trainer.best_model, trainer.feature_cols, X_test, y_test)
    importance_df = interpreter.analyze_feature_importance()
    interpreter.generate_business_insights(importance_df)
else:
    print("Model interpretability analysis requires tree-based models")

# ================================================================
# 10. DEPLOYMENT-READY MODULAR STRUCTURE
# ================================================================

# Create modular components for deployment

# class DataPipeline:
#     """
#     Production-ready data processing pipeline
#     """
    
#     def __init__(self):
#         self.cleaner = DataCleaner()
#         self.engineer = FeatureEngineer()
#         self.is_fitted = False
        
#     def fit_transform(self, df):
#         """
#         Fit pipeline on training data and transform
#         """
#         # Clean data
#         df_clean = self.cleaner.clean_data(df)
        
#         # Engineer features
#         df_features = self.engineer.engineer_features(df_clean)
        
#         self.is_fitted = True
#         return df_features
    
#     def transform(self, df):
#         """
#         Transform new data using fitted pipeline
#         """
#         if not self.is_fitted:
#             raise ValueError("Pipeline must be fitted before transform")
            
#         # Apply same transformations
#         df_clean = self.cleaner.clean_data(df)
#         df_features = self.engineer.engineer_features(df_clean)
        
#         return df_features

# class PricingModel:
#     """
#     Production-ready pricing model wrapper
#     """
    
#     def __init__(self):
#         self.model = None
#         self.pipeline = None
#         self.feature_cols = None
#         self.is_trained = False
        
#     def train(self,
#             (df['