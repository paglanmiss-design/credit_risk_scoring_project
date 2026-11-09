import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path: str) -> pd.DataFrame:
	return pd.read_csv(path)

def compute_and_save_correlation(df: pd.DataFrame, out_path: str):
	corr = df.corr()
	print("Correlation matrix:")
	print(corr)
	plt.figure(figsize=(8, 6))
	sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
	plt.title('Correlation matrix')
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()

def boxplot_by_default(df: pd.DataFrame, y_col: str, out_path: str, y_label: str):
	plt.figure(figsize=(6, 6))
	# Map default 0/1 to labels for clarity
	df_plot = df.copy()
	df_plot['default_label'] = df_plot['default'].map({0: 'Creditworthy', 1: 'High-risk'})
	sns.boxplot(x='default_label', y=y_col, data=df_plot)
	plt.xlabel('Client category')
	plt.ylabel(y_label)
	plt.title(f'{y_label} by client category')
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()

def main():
	data_path = os.path.join('data', 'clean_data.csv')
	graphs_dir = 'graphs'
	os.makedirs(graphs_dir, exist_ok=True)

	df = load_data(data_path)

	# Correlation matrix
	corr_out = os.path.join(graphs_dir, 'correlation_matrix.png')
	compute_and_save_correlation(df, corr_out)

	# Boxplots
	boxplot_by_default(df, 'debt_to_income_ratio', os.path.join(graphs_dir, 'boxplot_debt_to_income_ratio.png'), 'Debt-to-income ratio (%)')
	boxplot_by_default(df, 'income', os.path.join(graphs_dir, 'boxplot_income.png'), 'Monthly income')

	# Key business insight
	dti_by_default = df.groupby('default')['debt_to_income_ratio'].median()
	income_by_default = df.groupby('default')['income'].median()

	insight = (
		"Key business insight: Clients who default consistently show a higher debt-to-income ratio "
		f"(median DTI default=1: {dti_by_default.get(1, float('nan'))} vs default=0: {dti_by_default.get(0, float('nan'))}) "
		f"and lower income (median income default=1: {income_by_default.get(1, float('nan'))} vs default=0: {income_by_default.get(0, float('nan'))})."
	)
	print('\n' + insight)

if __name__ == '__main__':
	main()
