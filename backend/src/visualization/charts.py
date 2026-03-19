import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, Any

def generate_chart(df: pd.DataFrame, chart_type: str, columns: list = None) -> str:
    """Generate chart and return as base64 encoded string"""
    plt.figure(figsize=(10, 6))
    
    if chart_type == "histogram" and columns:
        df[columns[0]].hist(bins=30)
        plt.title(f"Distribution of {columns[0]}")
        plt.xlabel(columns[0])
        plt.ylabel("Frequency")
    
    elif chart_type == "correlation":
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title("Correlation Matrix")
    
    elif chart_type == "scatter" and len(columns) >= 2:
        plt.scatter(df[columns[0]], df[columns[1]], alpha=0.6)
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.title(f"{columns[0]} vs {columns[1]}")
    
    elif chart_type == "bar" and columns:
        value_counts = df[columns[0]].value_counts().head(10)
        value_counts.plot(kind='bar')
        plt.title(f"Top 10 values in {columns[0]}")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

def auto_visualize(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """Auto-generate appropriate visualization based on question"""
    try:
        question_lower = question.lower()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Check for specific column names in the question
        mentioned_cols = [col for col in df.columns if col.lower() in question_lower]
        
        # Look for relationship/scatter plot keywords
        if any(word in question_lower for word in ["relation", "relationship", "scatter", "plot", "vs", "between"]):
            if len(mentioned_cols) >= 2:
                # Use mentioned columns if available
                chart = generate_chart(df, "scatter", mentioned_cols[:2])
                return {"type": "scatter", "chart": chart, "description": f"Scatter plot of {mentioned_cols[0]} vs {mentioned_cols[1]}"}
            elif len(numeric_cols) >= 2:
                # Fallback to first two numeric columns
                chart = generate_chart(df, "scatter", numeric_cols[:2])
                return {"type": "scatter", "chart": chart, "description": f"Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}"}
        
        # Check for correlation
        if "correlation" in question_lower and len(numeric_cols) > 1:
            chart = generate_chart(df, "correlation")
            return {"type": "correlation", "chart": chart, "description": "Correlation matrix of numeric variables"}
        
        # Check for distribution
        if "distribution" in question_lower:
            target_col = mentioned_cols[0] if mentioned_cols else (numeric_cols[0] if numeric_cols else None)
            if target_col:
                chart = generate_chart(df, "histogram", [target_col])
                return {"type": "histogram", "chart": chart, "description": f"Distribution of {target_col}"}
        
        # Default fallback - create a scatter plot if we have numeric columns
        if len(numeric_cols) >= 2:
            chart = generate_chart(df, "scatter", numeric_cols[:2])
            return {"type": "scatter", "chart": chart, "description": f"Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}"}
        
        # If only categorical data, show bar chart
        if categorical_cols:
            chart = generate_chart(df, "bar", [categorical_cols[0]])
            return {"type": "bar", "chart": chart, "description": f"Bar chart of {categorical_cols[0]}"}
        
        # Last resort - correlation if we have any numeric data
        if len(numeric_cols) > 1:
            chart = generate_chart(df, "correlation")
            return {"type": "correlation", "chart": chart, "description": "Correlation matrix of numeric variables"}
            
        return None
        
    except Exception as e:
        print(f"Error in auto_visualize: {e}")
        return None