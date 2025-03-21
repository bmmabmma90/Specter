import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import os

# --- Streamlit Configuration ---
st.set_page_config(layout="wide")  # Make the window wide

# --- Data Loading and Cleaning ---

def load_and_clean_data(file_path):
    """Loads and cleans the data from the given file path."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df_original = df.copy()  # Store the original data
 
        relevant_columns = ['Rank', 'Company Name', 'Website', 'Founded Date', 'Total Funding Amount (in USD)',
                            'Employee Count', 'HQ Location', 'Growth Stage', 'Web Visits', 'LinkedIn - Followers',
                            'Twitter - Followers', 'Reported Traction Highlights', 'Investors' ]
        df_cleaned = df.copy()
        df_cleaned = df_cleaned[relevant_columns]

        df_cleaned['Total Funding Amount (in USD)'] = df_cleaned['Total Funding Amount (in USD)'].fillna(0)

        for col in ['Founded Date', 'Total Funding Amount (in USD)', 'Rank', 'Employee Count', 'Web Visits',
                    'LinkedIn - Followers', 'Twitter - Followers']:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype('Int64')
            df_cleaned[col] = df_cleaned[col].fillna(0)

        return df_cleaned, df_original

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None

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

def extract_last_part(location_str):
    """Extracts the last part of a string after the last comma."""
    if pd.isna(location_str) or not isinstance(location_str, str):
        return ""
    parts = location_str.split(',')
    if len(parts) > 0:
        return parts[-1].strip()
    return ""

def flexible_formatter(x, pos, max_value):
    """Formats numbers to display in millions or thousands."""
    if max_value >= 1000000:
        return f'{x / 1e6:.1f}M'
    elif max_value >= 1000:
        return f'{x / 1e3:.1f}K'
    else:
        return f'{x:.0f}'

# --- Data Display ---

def display_interactive_table(df):
    """Displays an interactive table with formatted numbers."""
    df_display = df.copy()
    df_display.sort_values(by='Total Funding Amount (in USD)', ascending=False, inplace=True)
    df_display['Total Funding Amount (in USD)'] = df_display['Total Funding Amount (in USD)'].apply(format_number)
    df_display['Web Visits'] = df_display['Web Visits'].apply(format_number)
    df_display["HQ Location"] = df_display["HQ Location"].apply(extract_last_part)

    df_display.rename(columns={
      'Founded Date': 'Founded',
      'Total Funding Amount (in USD)': 'Funding (USD)',
      'Employee Count': 'Employees',
      'LinkedIn - Followers' : 'LinkedIn',
      'Twitter - Followers' : 'Twitter'
    }, inplace=True)

    st.data_editor(
        df_display,
        column_config={
            "Website": st.column_config.LinkColumn("Website", width="medium"),
            "Founded": st.column_config.NumberColumn("Founded", format="%d"),
            "Funding (USD)": st.column_config.TextColumn("Funding (USD)"),
            "Employees": st.column_config.NumberColumn("Employees", format="%d"),
            "LinkedIn": st.column_config.NumberColumn("LinkedIn", format="%d"),
            "Twitter": st.column_config.NumberColumn("Twitter", format="%d"),
        },
        hide_index=True,
        disabled=df_display.columns
    )
# --- Plotting ---

def display_bubble_plot(df):
    """Displays the bubble plot."""
    final_data = df.copy()
    final_data = final_data[final_data['Total Funding Amount (in USD)'] != 0]
    max_funding = final_data['Total Funding Amount (in USD)'].max()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=final_data, x='Rank', y='Total Funding Amount (in USD)', size='Employee Count',
                    sizes=(20, 1000), hue='Web Visits', palette='viridis', alpha=0.6, legend=False, ax=ax)

    for i in range(final_data.shape[0]):
        ax.text(final_data['Rank'].iloc[i], final_data['Total Funding Amount (in USD)'].iloc[i],
                 f"{final_data['Company Name'].iloc[i]} ({final_data['Founded Date'].iloc[i]}) Visits:{final_data['Web Visits'].iloc[i]}",
                 fontsize=8)

    ax.set_title('Total Funding vs Rank with Bubble Sizes based on Employee Count')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Total Funding Amount')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: flexible_formatter(x, pos, max_funding)))
    st.pyplot(fig)

def plot_sorted_bar_chart(df, column_name, title, color):
    """Displays a sorted bar chart with relative percentages."""
    sorted_df = df.sort_values(by=column_name, ascending=False)
    max_value = sorted_df[column_name].max()

    sorted_df['Relative Percentage'] = (sorted_df[column_name] / max_value) * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(sorted_df['Company Name'], sorted_df[column_name], color=color)
    ax.set_title(title)
    ax.set_xlabel('Company Name')
    ax.set_ylabel(column_name)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: flexible_formatter(x, pos, max_value)))

    plt.xticks(rotation=90)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01 * max_value,
                 f'{sorted_df["Relative Percentage"].iloc[bars.index(bar)]:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

def display_correlation_graphs(df):
    """Displays various correlation graphs."""
    st.subheader("Correlation and Distribution Graphs")
    max_funding = df['Total Funding Amount (in USD)'].max()

    st.write("Funding Amount vs. Employee Count (Colored by Growth Stage)")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Employee Count', y='Total Funding Amount (in USD)', data=df, hue='Growth Stage', ax=ax1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: flexible_formatter(x, pos, max_funding)))
    ax1.set_ylabel('Total Funding Amount')
    st.pyplot(fig1)

    correlation = df['Employee Count'].corr(df['Total Funding Amount (in USD)'])
    st.write(f"Correlation between Employee Count and Total Funding Amount: {correlation}")

    st.write("Funding Amount vs. Employee Count with Regression Line")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.regplot(x='Employee Count', y='Total Funding Amount (in USD)', data=df, ax=ax2)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: flexible_formatter(x, pos, max_funding)))
    ax2.set_ylabel('Total Funding Amount')
    st.pyplot(fig2)

    st.write("Distribution of Total Funding Amount")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Total Funding Amount (in USD)'], kde=True, ax=ax3)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: flexible_formatter(x, pos, max_funding)))
    ax3.set_xlabel('Total Funding Amount')
    st.pyplot(fig3)

    st.write("Funding Amount Distribution by Growth Stage")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Growth Stage', y='Total Funding Amount (in USD)', data=df, ax=ax4)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: flexible_formatter(x, pos, max_funding)))
    ax4.set_ylabel('Total Funding Amount')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig4)

    st.write("Funding Amount vs. Web Visits")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Web Visits', y='Total Funding Amount (in USD)', data=df, ax=ax5)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: flexible_formatter(x, pos, max_funding)))
    ax5.set_ylabel('Total Funding Amount')
    st.pyplot(fig5)

# --- Company Data Display ---

def display_company_data(df):
    """Displays detailed data for a selected company in a two-column format."""
    company_names = sorted(df['Company Name'].unique().tolist())
    selected_company = st.selectbox('Select Company:', company_names)
    company_data = df[df['Company Name'] == selected_company].iloc[0]

    st.subheader(f"Data for {selected_company}:")

    # Create a DataFrame for the two-column table
    data_to_display = pd.DataFrame({
        'Attribute': company_data.index,
        'Value': company_data.values
    })
    st.table(data_to_display)

# --- Raw Data Display ---
def display_raw_data(df):
    """Displays detailed raw data for a selected company."""
    company_names = sorted(df['Company Name'].unique().tolist())
    selected_company = st.selectbox('Select Company:', company_names)
    company_data = df[df['Company Name'] == selected_company]

    st.subheader(f"Raw Data for {selected_company}:")

    for column in company_data.columns:
        st.write(f"**{column}:** ", company_data[column].iloc[0])

# --- Growth Data Display ---

def display_growth_data(df):
    """Displays data related to 'Growth'."""
    growth_columns = [col for col in df.columns if "Growth" in col]
    web_visits_columns = [col for col in df.columns if "Web Visits" in col]

    if not growth_columns and not web_visits_columns:
        st.write("No columns found containing 'Growth' or 'Web Visits' in their name.")
        return

    if growth_columns:
        growth_df = df[['Company Name', 'Website', 'Employee Count', 'LinkedIn - Followers', 'Twitter - Followers'] + growth_columns]
        grouped_growth_data = growth_df.groupby(['Website', 'Employee Count', 'LinkedIn - Followers', 'Twitter - Followers'])[growth_columns].agg(lambda x: ', '.join(map(str, x))).reset_index()
        st.dataframe(grouped_growth_data, hide_index=True)

    if web_visits_columns:
        web_visits_df = df[['Company Name', 'Website'] + web_visits_columns].copy()
        
        # Rename columns but remember to replace 7 with 12 and 8 with 24
        rename_dict = {'Web Visits': '0'}
        for i, col in enumerate(web_visits_columns):
            if i > 0:
                # if i == 7:
                #     rename_dict[col] = '12'
                # elif i == 8:
                #     rename_dict[col] = '24'
                # else:
                rename_dict[col] = str(i)
        web_visits_df.rename(columns=rename_dict, inplace=True)
        
        # Create 'data' row
        for index, row in web_visits_df.iterrows():
            base_value = row['0']
            for i in range(1, len(web_visits_columns)):
                growth_percentage = float(row[str(i)])/100
                if pd.isna(growth_percentage) or growth_percentage == 0.0:
                    web_visits_df.loc[index, str(i)] = 0.0
                else:
                    try:
                        web_visits_df.loc[index, str(i)] = base_value / (1 + growth_percentage)
                    except (TypeError, ZeroDivisionError):
                        web_visits_df.loc[index, str(i)] = 0.0
        st.subheader("Web Visits Data")
        st.dataframe(web_visits_df)
        # Line Graph
        st.subheader("Web Visits Line Graph")
        
        # Select only the columns with the calculated values
        line_graph_data = web_visits_df[[str(i) for i in range(len(web_visits_columns)) if str(i) in web_visits_df.columns or i == 0]]
        line_graph_data.index = web_visits_df['Company Name']
        
        # Transpose the DataFrame to have months as columns
        line_graph_data = line_graph_data.transpose()
        
        # Plot the line graph
        fig_line, ax_line = plt.subplots(figsize=(14, 6))
        for company in line_graph_data.columns:
            ax_line.plot(line_graph_data.index, line_graph_data[company], marker='o', label=company)
           
        ax_line.set_title('Web Visits Over Time')
        ax_line.set_xlabel('Month (Relative)')
        ax_line.set_ylabel('Web Visits')
        ax_line.legend()
        ax_line.grid(True)
        st.pyplot(fig_line)

# --- Main App ---

def main():
    """Main function for the Streamlit app."""
    st.title("Specter Data Analysis")

    # Get query parameters
    query_params = st.query_params
    test_mode = query_params.get("test", "false").lower() == "true"

    if test_mode:
        st.write("Running in test mode...")
        TEST_FILE_NAME = "specter-company-signals.csv"
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        file_path = os.path.join(downloads_path, TEST_FILE_NAME)
        st.write("Attempting to load", file_path)

        if os.path.exists(file_path):
            df = load_and_clean_data(file_path)
        else:
            st.error(f"Test file not found at: {file_path}")
            df = None

    else:
        st.write("Upload your data file and then select what data you want to see from the sidebar menu.")
        # File upload
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            # Directly use StringIO for the uploaded file
            df, df_original = load_and_clean_data(uploaded_file)
        else:
            df = None
            df_original = None

    if df is not None:
        # Sidebar menu
        menu = ["Interactive Table", "Bubble Plot", "Social Media Data", "Correlation Graphs", "Company Data", "Raw Data", "Growth Data"]
        choice = st.sidebar.selectbox("Select an option:", menu)

        if choice == "Interactive Table":
            st.subheader("Interactive Table")
            display_interactive_table(df)
        elif choice == "Bubble Plot":
            st.subheader("Bubble Plot")
            display_bubble_plot(df)
        elif choice == "Social Media Data":
            st.subheader("Social Media Data")
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
            display_growth_data(df_original)
        elif choice == "Raw Data":
            st.subheader("Raw Data")
            if df_original is not None:
                display_raw_data(df_original)
            else:
                st.error("Original Data not available")

if __name__ == "__main__":
    main()
