import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def create_comprehensive_dashboard(self) -> Dict[str, str]:
        """Create a comprehensive dashboard with multiple visualizations"""
        charts = {}
        
        # 1. Dataset Overview
        charts['overview'] = self._create_overview_chart()
        
        # 2. Missing Data Analysis
        if self.df.isnull().sum().sum() > 0:
            charts['missing_data'] = self._create_missing_data_chart()
        
        # 3. Numeric Variables Analysis
        if len(self.numeric_cols) > 0:
            charts['numeric_distributions'] = self._create_numeric_distributions()
            charts['correlation_matrix'] = self._create_correlation_matrix()
            
            if len(self.numeric_cols) >= 2:
                charts['pairplot'] = self._create_pairplot()
        
        # 4. Categorical Variables Analysis
        if len(self.categorical_cols) > 0:
            charts['categorical_analysis'] = self._create_categorical_analysis()
        
        # 5. Outlier Detection
        if len(self.numeric_cols) > 0:
            charts['outlier_detection'] = self._create_outlier_detection()
        
        # 6. Data Quality Assessment
        charts['data_quality'] = self._create_data_quality_chart()
        
        return charts
    
    def _create_overview_chart(self) -> str:
        """Create dataset overview visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Data types distribution
        dtype_counts = self.df.dtypes.value_counts()
        ax1.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Data Types Distribution')
        
        # 2. Missing data by column
        missing_data = self.df.isnull().sum().sort_values(ascending=False)
        if missing_data.sum() > 0:
            missing_data = missing_data[missing_data > 0][:10]  # Top 10
            ax2.barh(range(len(missing_data)), missing_data.values)
            ax2.set_yticks(range(len(missing_data)))
            ax2.set_yticklabels(missing_data.index)
            ax2.set_xlabel('Missing Values Count')
            ax2.set_title('Missing Data by Column')
        else:
            ax2.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Missing Data Status')
        
        # 3. Unique values distribution
        unique_counts = self.df.nunique().sort_values(ascending=False)[:10]
        ax3.bar(range(len(unique_counts)), unique_counts.values)
        ax3.set_xticks(range(len(unique_counts)))
        ax3.set_xticklabels(unique_counts.index, rotation=45, ha='right')
        ax3.set_ylabel('Unique Values Count')
        ax3.set_title('Unique Values by Column')
        
        # 4. Dataset statistics
        stats_text = f"""
        Rows: {self.df.shape[0]:,}
        Columns: {self.df.shape[1]}
        Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        Duplicates: {self.df.duplicated().sum():,}
        Missing Values: {self.df.isnull().sum().sum():,}
        Completeness: {(1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100:.1f}%
        """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12, verticalalignment='center')
        ax4.set_title('Dataset Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_missing_data_chart(self) -> str:
        """Create missing data visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Missing Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Missing data heatmap
        missing_data = self.df.isnull()
        if missing_data.sum().sum() > 0:
            sns.heatmap(missing_data, cbar=True, ax=ax1, cmap='viridis')
            ax1.set_title('Missing Data Heatmap')
            ax1.set_xlabel('Columns')
            ax1.set_ylabel('Rows (sample)')
        
        # 2. Missing data percentage by column
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        if len(missing_pct) > 0:
            ax2.barh(range(len(missing_pct)), missing_pct.values)
            ax2.set_yticks(range(len(missing_pct)))
            ax2.set_yticklabels(missing_pct.index)
            ax2.set_xlabel('Missing Percentage (%)')
            ax2.set_title('Missing Data Percentage by Column')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_numeric_distributions(self) -> str:
        """Create distributions for numeric variables"""
        n_cols = min(len(self.numeric_cols), 6)  # Limit to 6 for readability
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Numeric Variables Distributions', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i, col in enumerate(self.numeric_cols[:n_cols]):
            data = self.df[col].dropna()
            
            # Histogram with KDE
            axes[i].hist(data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Add KDE curve
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except:
                pass
            
            axes[i].set_title(f'{col}\nMean: {data.mean():.2f}, Std: {data.std():.2f}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_cols, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_correlation_matrix(self) -> str:
        """Create correlation matrix heatmap"""
        if len(self.numeric_cols) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_pairplot(self) -> str:
        """Create pairplot for numeric variables"""
        if len(self.numeric_cols) < 2:
            return None
        
        # Limit to first 5 numeric columns for performance
        cols_to_plot = self.numeric_cols[:5]
        
        # Create pairplot
        g = sns.pairplot(self.df[cols_to_plot], diag_kind='hist', plot_kws={'alpha': 0.6})
        g.fig.suptitle('Pairwise Relationships', y=1.02, fontsize=16, fontweight='bold')
        
        return self._fig_to_base64(g.fig)
    
    def _create_categorical_analysis(self) -> str:
        """Create categorical variables analysis"""
        n_cols = min(len(self.categorical_cols), 4)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Categorical Variables Analysis', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i, col in enumerate(self.categorical_cols[:n_cols]):
            value_counts = self.df[col].value_counts().head(10)
            
            axes[i].bar(range(len(value_counts)), value_counts.values, color='lightcoral')
            axes[i].set_xticks(range(len(value_counts)))
            axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[i].set_title(f'{col} (Top 10 Values)')
            axes[i].set_ylabel('Count')
            
            # Add value labels on bars
            for j, v in enumerate(value_counts.values):
                axes[i].text(j, v + max(value_counts.values) * 0.01, str(v), ha='center')
        
        # Hide unused subplots
        for i in range(n_cols, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_outlier_detection(self) -> str:
        """Create outlier detection visualization"""
        if len(self.numeric_cols) == 0:
            return None
        
        n_cols = min(len(self.numeric_cols), 6)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Outlier Detection (Box Plots)', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i, col in enumerate(self.numeric_cols[:n_cols]):
            data = self.df[col].dropna()
            
            # Box plot
            box_plot = axes[i].boxplot(data, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            
            # Calculate outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
            
            axes[i].set_title(f'{col}\nOutliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_cols, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_data_quality_chart(self) -> str:
        """Create data quality assessment visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Quality Assessment', fontsize=16, fontweight='bold')
        
        # 1. Completeness by column
        completeness = (1 - self.df.isnull().sum() / len(self.df)) * 100
        colors = ['red' if x < 80 else 'orange' if x < 95 else 'green' for x in completeness]
        ax1.barh(range(len(completeness)), completeness.values, color=colors)
        ax1.set_yticks(range(len(completeness)))
        ax1.set_yticklabels(completeness.index)
        ax1.set_xlabel('Completeness (%)')
        ax1.set_title('Data Completeness by Column')
        ax1.axvline(x=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        ax1.axvline(x=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
        ax1.legend()
        
        # 2. Data types distribution
        dtype_counts = self.df.dtypes.value_counts()
        ax2.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Data Types Distribution')
        
        # 3. Unique values ratio
        unique_ratio = self.df.nunique() / len(self.df) * 100
        ax3.bar(range(len(unique_ratio)), unique_ratio.values, color='lightgreen')
        ax3.set_xticks(range(len(unique_ratio)))
        ax3.set_xticklabels(unique_ratio.index, rotation=45, ha='right')
        ax3.set_ylabel('Unique Values Ratio (%)')
        ax3.set_title('Uniqueness by Column')
        
        # 4. Quality score summary
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        duplicate_rows = self.df.duplicated().sum()
        
        quality_metrics = {
            'Completeness': (1 - missing_cells / total_cells) * 100,
            'Uniqueness': (1 - duplicate_rows / len(self.df)) * 100,
            'Consistency': 95,  # Placeholder - would need more complex logic
            'Validity': 90     # Placeholder - would need domain-specific rules
        }
        
        metrics_names = list(quality_metrics.keys())
        metrics_values = list(quality_metrics.values())
        colors = ['red' if x < 70 else 'orange' if x < 85 else 'green' for x in metrics_values]
        
        ax4.bar(metrics_names, metrics_values, color=colors)
        ax4.set_ylabel('Score (%)')
        ax4.set_title('Data Quality Metrics')
        ax4.set_ylim(0, 100)
        
        # Add value labels on bars
        for i, v in enumerate(metrics_values):
            ax4.text(i, v + 2, f'{v:.1f}%', ha='center')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def create_custom_visualization(self, chart_type: str, columns: List[str] = None, question: str = "") -> str:
        """Create custom visualization based on user request"""
        if chart_type == "scatter" and columns and len(columns) >= 2:
            return self._create_scatter_plot(columns[0], columns[1])
        elif chart_type == "histogram" and columns:
            return self._create_histogram(columns[0])
        elif chart_type == "correlation":
            return self._create_correlation_matrix()
        elif chart_type == "bar" and columns:
            return self._create_bar_chart(columns[0])
        elif chart_type == "box" and columns:
            return self._create_box_plot(columns[0])
        else:
            return self._auto_visualize_from_question(question)
    
    def _create_scatter_plot(self, x_col: str, y_col: str) -> str:
        """Create enhanced scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        scatter = ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6, s=50, color='blue')
        
        # Add trend line
        try:
            z = np.polyfit(self.df[x_col].dropna(), self.df[y_col].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(self.df[x_col], p(self.df[x_col]), "r--", alpha=0.8, linewidth=2)
        except:
            pass
        
        # Calculate correlation
        corr = self.df[x_col].corr(self.df[y_col])
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{x_col} vs {y_col}\nCorrelation: {corr:.3f}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_histogram(self, col: str) -> str:
        """Create enhanced histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = self.df[col].dropna()
        
        # Create histogram
        n, bins, patches = ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {col}\nStd: {std_val:.2f}, Skewness: {data.skew():.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_bar_chart(self, col: str) -> str:
        """Create enhanced bar chart"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        value_counts = self.df[col].value_counts().head(15)
        
        bars = ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {col} (Top 15 Values)')
        
        # Add value labels on bars
        for i, v in enumerate(value_counts.values):
            ax.text(i, v + max(value_counts.values) * 0.01, str(v), ha='center')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_box_plot(self, col: str) -> str:
        """Create enhanced box plot"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data = self.df[col].dropna()
        box_plot = ax.boxplot(data, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        
        # Add statistics
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
        
        ax.set_ylabel(col)
        ax.set_title(f'Box Plot of {col}\nOutliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _auto_visualize_from_question(self, question: str) -> str:
        """Auto-generate visualization based on question"""
        question_lower = question.lower()
        
        # Extract column names mentioned in question
        mentioned_cols = [col for col in self.df.columns if col.lower() in question_lower]
        
        # Determine chart type based on keywords
        if any(word in question_lower for word in ["relation", "relationship", "scatter", "vs", "between"]):
            if len(mentioned_cols) >= 2:
                return self._create_scatter_plot(mentioned_cols[0], mentioned_cols[1])
            elif len(self.numeric_cols) >= 2:
                return self._create_scatter_plot(self.numeric_cols[0], self.numeric_cols[1])
        
        elif "distribution" in question_lower or "histogram" in question_lower:
            if mentioned_cols:
                return self._create_histogram(mentioned_cols[0])
            elif self.numeric_cols:
                return self._create_histogram(self.numeric_cols[0])
        
        elif "correlation" in question_lower:
            return self._create_correlation_matrix()
        
        # Default: return overview
        return self._create_overview_chart()
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64

# Convenience functions for backward compatibility
def generate_chart(df: pd.DataFrame, chart_type: str, columns: list = None) -> str:
    """Generate chart and return as base64 encoded string"""
    visualizer = AdvancedVisualizer(df)
    return visualizer.create_custom_visualization(chart_type, columns)

def auto_visualize(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """Auto-generate appropriate visualization based on question"""
    visualizer = AdvancedVisualizer(df)
    
    try:
        chart = visualizer._auto_visualize_from_question(question)
        if chart:
            return {
                "type": "auto",
                "chart": chart,
                "description": f"Visualization generated for: {question}"
            }
    except Exception as e:
        print(f"Error in auto_visualize: {e}")
    
    return None