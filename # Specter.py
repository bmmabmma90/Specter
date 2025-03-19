import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# --- Data Loading and Cleaning ---

def load_and_clean_data(uploaded_file):
    """Loads and cleans the data from the uploaded file."""
    if uploaded_file is not None:
        try:
            # Read the CSV content from the file-like object
            csv_content = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(csv_content, encoding='utf-8')

            relevant_columns = ['Rank', 'Company Name', 'Website', 'Founded Date', 'Total Funding Amount (in USD)',
                                'Employee Count', 'Web Visits', 'HQ Location', 'Growth Stage',
                                'Reported Traction Highlights', 'Investors', 'LinkedIn - Followers',
                                'Twitter - Followers']
            df_cleaned = df.copy()
            df_cleaned = df_cleaned[relevant_columns]

            df_cleaned['Total Funding Amount (in USD)'] = df_cleaned['Total Funding Amount (in USD)'].fillna(0)

            for col in ['Founded Date', 'Total Funding Amount (in USD)', 'Rank', 'Employee Count', 'Web Visits',
                        'LinkedIn - Followers', 'Twitter - Followers']:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype('Int64')
                df_cleaned[col] = df_cleaned[col].fillna(0)

            return df_cleaned

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return None
    else:
        return None

# --- Data Formatting ---

def format_number(num):
    """Formats numbers with thousands separators and K/M suffixes."""
    if pd.isna(num):
        return ""
    num = int(num)
    if num >= 1000000:
        return f"{num / 1000000:.1f}M"
    elif num >= 1000:
        return f"{num / 1000:.1f}K"
    else:
        return str(num)

# --- Data Display ---

def display_interactive_table(df):
    """Displays an interactive table with formatted numbers."""
    df_display = df.copy()
    df_display.sort_values(by='Total Funding Amount (in USD)', ascending=False, inplace=True)
    df_display['Total Funding Amount (in USD)'] = df_display['Total Funding Amount (in USD)'].apply(format_number)
    df_display['Web Visits'] = df_display['Web Visits'].apply(format_number)
    st.dataframe(df_display, hide_index=True)
    
# --- Plotting ---

def display_bubble_plot(df):
    """Displays the bubble plot."""
    final_data = df.copy()
    final_data = final_data[final_data['Total Funding Amount (in USD)'] != 0]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=final_data, x='Rank', y='Total Funding Amount (in USD)', size='Employee Count',
                    sizes=(20, 1000), hue='Web Visits', palette='viridis', alpha=0.6, legend=False)

    for i in range(final_data.shape[0]):
        plt.text(final_data['Rank'].iloc[i], final_data['Total Funding Amount (in USD)'].iloc[i],
                 f"{final_data['Company Name'].iloc[i]} ({final_data['Founded Date'].iloc[i]}) Visits:{final_data['Web Visits'].iloc[i]}",
                 fontsize=8)

    plt.title('Total Funding vs Rank with Bubble Sizes based on Employee Count')
    plt.xlabel('Rank')
    plt.ylabel('Total Funding Amount (in USD)')
    plt.grid(True)
    st.pyplot(plt)

def plot_sorted_bar_chart(df, column_name, title, color):
    """Displays a sorted bar chart with relative percentages."""
    sorted_df = df.sort_values(by=column_name, ascending=False)
    max_value = sorted_df[column_name].max()
    sorted_df['Relative Percentage'] = (sorted_df[column_name] / max_value) * 100

    plt.figure(figsize=(14, 6))
    bars = plt.bar(sorted_df['Company Name'], sorted_df[column_name], color=color)
    plt.title(title)
    plt.xlabel('Company Name')
    plt.ylabel(column_name)
    plt.xticks(rotation=90)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01 * max_value,
                 f'{sorted_df["Relative Percentage"].iloc[bars.index(bar)]:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    st.pyplot(plt)

def display_correlation_graphs(df):
    """Displays various correlation graphs."""
    st.subheader("Correlation and Distribution Graphs")

    st.write("Funding Amount vs. Employee Count (Colored by Growth Stage)")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Employee Count', y='Total Funding Amount (in USD)', data=df, hue='Growth Stage', ax=ax1)
    st.pyplot(fig1)

    correlation = df['Employee Count'].corr(df['Total Funding Amount (in USD)'])
    st.write(f"Correlation between Employee Count and Total Funding Amount: {correlation}")

    st.write("Funding Amount vs. Employee Count with Regression Line")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.regplot(x='Employee Count', y='Total Funding Amount (in USD)', data=df, ax=ax2)
    st.pyplot(fig2)

    st.write("Distribution of Total Funding Amount")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Total Funding Amount (in USD)'], kde=True, ax=ax3)
    st.pyplot(fig3)

    st.write("Funding Amount Distribution by Growth Stage")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Growth Stage', y='Total Funding Amount (in USD)', data=df, ax=ax4)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig4)

    st.write("Funding Amount vs. Web Visits")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Web Visits', y='Total Funding Amount (in USD)', data=df, ax=ax5)
    st.pyplot(fig5)

# --- Company Data Display ---

def display_company_data(df):
    """Displays detailed data for a selected company."""
    company_names = df['Company Name'].unique().tolist()
    selected_company = st.selectbox('Select Company:', company_names)
    company_data = df[df['Company Name'] == selected_company]

    st.subheader(f"Data for {selected_company}:")

    for column in company_data.columns:
        st.write(f"**{column}:**")
        st.write(company_data[column].iloc[0])

# --- Growth Data Display ---

def display_growth_data(df):
    """Displays data related to 'Growth'."""
    growth_columns = [col for col in df.columns if "Growth" in col]

    if not growth_columns:
        st.write("No columns found containing 'Growth' in their name.")
    else:
        growth_df = df[['Company Name', 'Website', 'Employee Count', 'LinkedIn - Followers', 'Twitter - Followers'] + growth_columns]
        grouped_growth_data = growth_df.groupby(['Website', 'Employee Count', 'LinkedIn - Followers', 'Twitter - Followers'])[growth_columns].agg(lambda x: ', '.join(map(str, x))).reset_index()
        st.dataframe(grouped_growth_data, hide_index=True)

# --- Main App ---

def main():
    """Main function for the Streamlit app."""
    st.title("Specter Data Analysis")
    st.write("Upload your data file and then select what data you want to see from the sidebar menu.")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    df = load_and_clean_data(uploaded_file)

    if df is not None:
        # Sidebar menu
        menu = ["Interactive Table", "Bubble Plot", "Sorted Bar Charts", "Correlation Graphs", "Company Data", "Growth Data"]
        choice = st.sidebar.selectbox("Select an option:", menu)

        if choice == "Interactive Table":
            st.subheader("Interactive Table")
            display_interactive_table(df)
        elif choice == "Bubble Plot":
            st.subheader("Bubble Plot")
            display_bubble_plot(df)
        elif choice == "Sorted Bar Charts":
            st.subheader("Sorted Bar Charts")
            plot_sorted_bar_chart(df, 'Web Visits', 'Web Visits by Company (Relative Percentage)', 'green')
            plot_sorted_bar_chart(df, 'LinkedIn - Followers', 'LinkedIn Followers by Company (Relative Percentage)', 'blue')
            plot_sorted_bar_chart(df, 'Twitter - Followers', 'Twitter Followers by Company (Relative Percentage)', 'orange')
        elif choice == "Correlation Graphs":
            display_correlation_graphs(df)
        elif choice == "Company Data":
            st.subheader("Company Data")
            display_company_data(df)
        elif choice == "Growth Data":
            st.subheader("Growth Data")
            display_growth_data(df)

if __name__ == "__main__":
    main()
