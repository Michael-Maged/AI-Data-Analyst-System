import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

def get_comprehensive_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive dataset analysis with advanced statistics"""
    
    # Basic info
    basic_info = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(df.isnull().sum().sum()),
        "missing_percentage": round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2)
    }
    
    # Column analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    column_analysis = {}
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "missing_percentage": round(df[col].isnull().sum() / len(df) * 100, 2),
            "unique_count": int(df[col].nunique()),
            "unique_percentage": round(df[col].nunique() / len(df) * 100, 2)
        }
        
        if col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                col_info.update({
                    "type": "numeric",
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "skewness": float(stats.skew(col_data)),
                    "kurtosis": float(stats.kurtosis(col_data)),
                    "q25": float(col_data.quantile(0.25)),
                    "q75": float(col_data.quantile(0.75)),
                    "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                    "outliers_count": int(len(detect_outliers(col_data))),
                    "zero_count": int((col_data == 0).sum()),
                    "negative_count": int((col_data < 0).sum())
                })
        
        elif col in categorical_cols:
            col_info.update({
                "type": "categorical",
                "mode": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                "top_values": df[col].value_counts().head(5).to_dict(),
                "entropy": calculate_entropy(df[col])
            })
        
        elif col in datetime_cols:
            col_info.update({
                "type": "datetime",
                "min_date": str(df[col].min()),
                "max_date": str(df[col].max()),
                "date_range_days": (df[col].max() - df[col].min()).days
            })
        
        column_analysis[col] = col_info
    
    # Correlation analysis
    correlations = {}
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        correlations = {
            "matrix": corr_matrix.round(3).to_dict(),
            "high_correlations": find_high_correlations(corr_matrix),
            "multicollinearity_pairs": find_multicollinear_pairs(corr_matrix)
        }
    
    # Data quality assessment
    data_quality = assess_data_quality(df)
    
    # Statistical insights
    insights = generate_statistical_insights(df, column_analysis)
    
    return {
        "basic_info": basic_info,
        "column_analysis": column_analysis,
        "correlations": correlations,
        "data_quality": data_quality,
        "insights": insights,
        "recommendations": generate_recommendations(df, column_analysis, data_quality)
    }

def detect_outliers(series: pd.Series) -> List[int]:
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)].index.tolist()

def calculate_entropy(series: pd.Series) -> float:
    """Calculate entropy for categorical data"""
    value_counts = series.value_counts()
    probabilities = value_counts / len(series)
    entropy = -sum(probabilities * np.log2(probabilities + 1e-10))
    return round(entropy, 3)

def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
    """Find pairs of variables with high correlation"""
    high_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corrs.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "correlation": round(corr_val, 3),
                    "strength": "strong" if abs(corr_val) >= 0.8 else "moderate"
                })
    return high_corrs

def find_multicollinear_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.9) -> List[Dict]:
    """Find multicollinear variable pairs"""
    multicollinear = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                multicollinear.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "correlation": round(corr_matrix.iloc[i, j], 3)
                })
    return multicollinear

def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Assess overall data quality"""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    
    quality_score = max(0, 100 - (missing_cells / total_cells * 100))
    
    issues = []
    if df.duplicated().sum() > 0:
        issues.append(f"{df.duplicated().sum()} duplicate rows found")
    
    high_missing_cols = df.columns[df.isnull().sum() / len(df) > 0.5].tolist()
    if high_missing_cols:
        issues.append(f"Columns with >50% missing: {high_missing_cols}")
    
    return {
        "quality_score": round(quality_score, 1),
        "completeness": round((1 - missing_cells / total_cells) * 100, 1),
        "issues": issues,
        "recommendations": []
    }

def generate_statistical_insights(df: pd.DataFrame, column_analysis: Dict) -> List[str]:
    """Generate key statistical insights"""
    insights = []
    
    # Dataset size insights
    if df.shape[0] > 100000:
        insights.append(f"Large dataset with {df.shape[0]:,} rows - suitable for machine learning")
    elif df.shape[0] < 100:
        insights.append(f"Small dataset with {df.shape[0]} rows - may need more data for robust analysis")
    
    # Missing data insights
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    if missing_pct > 20:
        insights.append(f"High missing data ({missing_pct:.1f}%) - consider imputation strategies")
    elif missing_pct < 5:
        insights.append("Low missing data - dataset is relatively complete")
    
    # Numeric insights
    numeric_cols = [col for col, info in column_analysis.items() if info.get("type") == "numeric"]
    if numeric_cols:
        skewed_cols = [col for col in numeric_cols if abs(column_analysis[col].get("skewness", 0)) > 1]
        if skewed_cols:
            insights.append(f"Highly skewed columns detected: {skewed_cols} - consider transformation")
    
    # Categorical insights
    categorical_cols = [col for col, info in column_analysis.items() if info.get("type") == "categorical"]
    high_cardinality = [col for col in categorical_cols if column_analysis[col]["unique_count"] > 50]
    if high_cardinality:
        insights.append(f"High cardinality categorical columns: {high_cardinality}")
    
    return insights

def generate_recommendations(df: pd.DataFrame, column_analysis: Dict, data_quality: Dict) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    if data_quality["quality_score"] < 80:
        recommendations.append("Consider data cleaning - quality score is below 80%")
    
    numeric_cols = [col for col, info in column_analysis.items() if info.get("type") == "numeric"]
    if len(numeric_cols) >= 2:
        recommendations.append("Perform correlation analysis to identify relationships between numeric variables")
    
    if df.duplicated().sum() > 0:
        recommendations.append("Remove duplicate rows to improve data quality")
    
    outlier_cols = [col for col, info in column_analysis.items() 
                   if info.get("type") == "numeric" and info.get("outliers_count", 0) > 0]
    if outlier_cols:
        recommendations.append(f"Investigate outliers in: {outlier_cols}")
    
    return recommendations

# Keep the original function for backward compatibility
def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Enhanced summary function - now uses comprehensive analysis"""
    comprehensive = get_comprehensive_summary(df)
    numeric = df.select_dtypes(include="number")

    return {
        "shape": comprehensive["basic_info"]["shape"],
        "columns": comprehensive["column_analysis"],
        "numeric_stats": numeric.describe().round(3).to_dict() if not numeric.empty else {},
        "correlations": comprehensive["correlations"].get("matrix", {}) if comprehensive["correlations"] else {},
        "top_categoricals": {
            col: info.get("top_values", {})
            for col, info in comprehensive["column_analysis"].items()
            if info.get("type") == "categorical"
        },
        "sample_rows": df.head(5).to_dict(orient="records"),
        "comprehensive": comprehensive
    }