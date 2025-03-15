import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class DatasetVisualizer:
    def __init__(self, csv_path=None):
        if csv_path is None:
            # Use the default path
            csv_path = os.path.join(os.path.dirname(__file__), '../model_accuracy_comparison.csv')
        
        # Read the CSV file
        self.df = pd.read_csv(csv_path)
        
        # Set the Model column as the index
        self.df_indexed = self.df.set_index('Model')
        
        # Remove the 'Overall' column for the dataset visualizations
        self.df_data_only = self.df_indexed.drop(columns=['Overall'])
        
        # Get dataset columns (excluding 'Overall')
        self.dataset_columns = [col for col in self.df.columns if col != 'Model' and col != 'Overall']
        
        # Create color palettes
        self.num_models = len(self.df)
        self.model_colors = plt.cm.viridis(np.linspace(0, 1, self.num_models))
        self.dataset_colors = plt.cm.plasma(np.linspace(0, 1, len(self.dataset_columns)))
    
    def create_stacked_by_dataset(self, save_path='stacked_by_dataset.png'):
        plt.figure(figsize=(15, 10))
        
        # Create positions for the bars
        bar_positions = np.arange(len(self.dataset_columns))
        bar_width = 0.8
        
        # Find the best performing model for each dataset
        best_models_per_dataset = {}
        for dataset in self.dataset_columns:
            best_model = self.df_indexed[dataset].idxmax()
            best_models_per_dataset[dataset] = best_model
        
        # Initialize the bottom of each bar to be 0
        bottom = np.zeros(len(self.dataset_columns))
        bar_objects = []
        
        # Plot each model as a segment in the stacked bars
        for i, model in enumerate(self.df['Model']):
            values = self.df_data_only.loc[model].values
            
            # Highlight the best model for each dataset with a black edge
            edge_colors = []
            line_widths = []
            for j, dataset in enumerate(self.dataset_columns):
                if best_models_per_dataset[dataset] == model:
                    edge_colors.append('black')
                    line_widths.append(2)
                else:
                    edge_colors.append('none')
                    line_widths.append(0)
            
            bars = plt.bar(bar_positions, values, bottom=bottom, width=bar_width, 
                          label=model, color=self.model_colors[i], alpha=0.8,
                          edgecolor=edge_colors, linewidth=line_widths)
            bar_objects.append(bars)
            bottom += values
        
        # Customize the plot
        plt.xlabel('Datasets', fontsize=12)
        plt.ylabel('Cumulative Accuracy', fontsize=12)
        plt.title('Stacked Accuracies by Dataset', fontsize=16)
        plt.xticks(bar_positions, self.dataset_columns, rotation=45)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        
        # Add text to explain the highlight
        plt.figtext(0.5, 0.01, "Note: Black outline indicates highest performing model for each dataset", 
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        # Save the visualization
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_stacked_by_model(self, save_path='stacked_by_model.png'):
        plt.figure(figsize=(15, 10))
        
        # Transpose the data to get datasets as rows and models as columns
        df_transposed = self.df_data_only.T
        
        # Create positions for the bars
        bar_positions = np.arange(len(df_transposed.columns))
        bar_width = 0.8
        
        # Find the best performing dataset for each model
        best_datasets_per_model = {}
        for model in self.df_indexed.index:
            best_dataset = self.df_data_only.loc[model].idxmax()
            best_datasets_per_model[model] = best_dataset
        
        # Initialize the bottom of each bar to be 0
        bottom = np.zeros(len(df_transposed.columns))
        
        # Plot each dataset as a segment in the stacked bars
        for i, dataset in enumerate(self.dataset_columns):
            values = df_transposed.loc[dataset].values
            
            # Highlight the best dataset for each model with a black edge
            edge_colors = []
            line_widths = []
            for j, model in enumerate(self.df_indexed.index):
                if best_datasets_per_model[model] == dataset:
                    edge_colors.append('black')
                    line_widths.append(2)
                else:
                    edge_colors.append('none')
                    line_widths.append(0)
            
            plt.bar(bar_positions, values, bottom=bottom, width=bar_width, 
                    label=dataset, color=self.dataset_colors[i], alpha=0.8,
                    edgecolor=edge_colors, linewidth=line_widths)
            bottom += values
        
        # Customize the plot
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Cumulative Accuracy', fontsize=12)
        plt.title('Stacked Dataset Accuracies by Model', fontsize=16)
        plt.xticks(bar_positions, df_transposed.columns, rotation=90)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        
        # Add text to explain the highlight
        plt.figtext(0.5, 0.01, "Note: Black outline indicates highest performing dataset for each model", 
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        # Save the visualization
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_overall_comparison(self, save_path='overall_accuracy.png'):
        plt.figure(figsize=(12, 6))
        
        overall_scores = self.df_indexed['Overall'].sort_values(ascending=False)
        bars = plt.bar(overall_scores.index, overall_scores.values, color=plt.cm.viridis(np.linspace(0, 1, len(overall_scores))))
        
        # Highlight the best performing model
        max_model = overall_scores.idxmax()
        for i, bar in enumerate(bars):
            if overall_scores.index[i] == max_model:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
        
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Overall Accuracy', fontsize=12)
        plt.title('Overall Model Accuracy Comparison', fontsize=16)
        plt.xticks(rotation=90)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom')
        
        # Save the visualization
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

    def create_type_accuracy_graph(self, csv_path=None, save_path='type_accuracy_stacked.png'):
        """
        Create a stacked bar graph showing model accuracy by data type.
        
        Args:
            csv_path (str, optional): Path to the CSV file containing type accuracy data.
                                     Defaults to '../type_accuracy.csv'.
            save_path (str, optional): Path to save the visualization.
                                      Defaults to 'type_accuracy_stacked.png'.
        
        Returns:
            str: Path where the visualization was saved.
        """
        if csv_path is None:
            # Use the default path
            csv_path = os.path.join(os.path.dirname(__file__), '../type_accuracy.csv')
        
        # Read the type accuracy CSV file
        df_type = pd.read_csv(csv_path)
        
        # Set the Model column as the index and sort by overall accuracy
        df_sorted = df_type.set_index('Model').sort_values('overall', ascending=False)
        
        # Get data type columns (excluding 'overall')
        data_types = [col for col in df_sorted.columns if col != 'overall']
        
        # Create the figure
        plt.figure(figsize=(15, 8))
        
        # Create distinct colors for each data type - use tab10 for highly distinct colors
        type_colors = plt.cm.tab10(np.linspace(0, 1, len(data_types)))
        
        # Create positions for the bars
        bar_positions = np.arange(len(df_sorted.index))
        bar_width = 0.65
        
        # Initialize the bottom of each bar to be 0
        bottom = np.zeros(len(df_sorted.index))
        
        # Plot each data type as a segment in the stacked bars
        for i, data_type in enumerate(data_types):
            values = df_sorted[data_type]
            plt.bar(bar_positions, values, bottom=bottom, width=bar_width,
                    label=data_type, color=type_colors[i])
            bottom += values
        
        # Customize the plot
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Cumulative Accuracy Across Data Types', fontsize=12)
        plt.title('Model Performance by Data Type', fontsize=16)
        plt.xticks(bar_positions, df_sorted.index, rotation=90)
        plt.legend(title='Data Types', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add overall accuracy markers
        for i, model in enumerate(df_sorted.index):
            overall = df_sorted.loc[model, 'overall']
            # Add a marker at the overall accuracy level
            plt.plot(i, overall, 'o', color='black', markersize=8)
            plt.text(i, overall + 0.1, f'Overall: {overall:.2f}', ha='center', 
                    fontsize=9, fontweight='bold')
        
        # Add text to explain the marker
        plt.figtext(0.5, 0.01, "Note: Black dots indicate overall accuracy for each model",
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        # Save the visualization
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

# Example usage
if __name__ == "__main__":
    visualizer = DatasetVisualizer()
    # visualizer.create_stacked_by_dataset()
    # visualizer.create_stacked_by_model()
    # visualizer.create_overall_comparison()
    visualizer.create_type_accuracy_graph()
    print("Visualizations have been created and saved as PNG files.")

