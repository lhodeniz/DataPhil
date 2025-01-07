# import modules

import pandas as pd
import streamlit as st
import numpy as np
import os
import json
from io import StringIO
import base64
from io import BytesIO
import zipfile
import openpyxl
import xlrd
import odf
import pyreadstat
import scipy
import pyreadr
import seaborn as sns 
import plotly.express as px
import textwrap
import matplotlib.pyplot as plt
import altair as alt
import datetime
import plotly.graph_objects as go
from wordcloud import WordCloud


# page config
st.set_page_config(page_title="DataPhil", layout="wide")

show_only_dashboard = st.toggle("Show Only Dashboard")


# initializations

if "section_selection" not in st.session_state:
    st.session_state.section_selection = "Upload Dataset"

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()


if 'selected_df' not in st.session_state:
      st.session_state.selected_df = pd.DataFrame()

if 'new_columns' not in st.session_state:
    st.session_state.new_columns = []

if 'original_columns' not in st.session_state:
 st.session_state.original_columns = st.session_state.df.columns.tolist() if not st.session_state.df.empty else []


if 'columns_to_delete' not in st.session_state:
 st.session_state.columns_to_delete = []

if "layout" not in st.session_state:
    st.session_state.layout = {}

if "charts" not in st.session_state:
    st.session_state.charts = {}

if 'tb' not in st.session_state:
    st.session_state.tb = {}

if 'filters' not in st.session_state:
    st.session_state.filters = []

if 'json_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'custom_functions' not in st.session_state:
    st.session_state.custom_functions = {}


if "custom_title" not in st.session_state:
    st.session_state.custom_title="Dashboard"

if "chart_type" not in st.session_state:
    st.session_state["chart_type"] = None  # Default to None


if 'agg_list' not in st.session_state:
    st.session_state.agg_list = []

if 'agg_event' not in st.session_state:
    st.session_state.agg_event = False

if 'agg_code' not in st.session_state:
    st.session_state.agg_code = ''

if 'chart_code' not in st.session_state:
    st.session_state.chart_code =''

def add_or_update_function():
    func_name = st.session_state.function_name
    func_code = st.session_state.function_code
    if func_name and func_code:
        st.session_state.custom_functions[func_name] = func_code
        st.success(f"Function '{func_name}' added/updated successfully!")

# Function to remove a custom function
def remove_function():
    func_name = st.session_state.function_to_remove
    if func_name in st.session_state.custom_functions:
        del st.session_state.custom_functions[func_name]
        st.success(f"Function '{func_name}' removed successfully!")



def backup_df():
    if 'df_backup' not in st.session_state or not st.session_state.df_backup.equals(st.session_state.df):
        st.session_state.df_backup = st.session_state.df.copy()

def restore_backup():
    if 'df_backup' in st.session_state:
        st.session_state.df = st.session_state.df_backup.copy()



# functions

def summary():
    # Check if the dataframe is empty
    if st.session_state.df.empty:
        st.warning("No dataset uploaded. Please upload a dataset to view the summary.")
        return  # Exit the function if no data is present

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Shape", "Data Types", "Numerical Data", "Non-Numerical Data", "Missing Values", "Duplicated Rows"])

    with tab1:
        st.markdown(f"Rows: {st.session_state.df.shape[0]:,}  \nColumns: {st.session_state.df.shape[1]:,}")

    with tab2:
        df_dtypes = st.session_state.df.dtypes.reset_index()
        df_dtypes.columns = ['Column', 'Type']
        df_dtypes.index = df_dtypes.index + 1
        st.write(df_dtypes)

    with tab3:
        st.write(st.session_state.df.describe())

    with tab4:
        # Check if there are columns of type 'object'
        if st.session_state.df.select_dtypes(include='object').shape[1] > 0:
            st.write(st.session_state.df.describe(include='object'))
        else:
            st.write("No non-numerical data columns to describe.")

    with tab5:
        df_nulls = st.session_state.df.isnull().sum().reset_index()
        df_nulls.columns = ['Column', 'Number of Null Values']
        df_nulls.index = df_nulls.index + 1
        st.write(df_nulls)

    with tab6:
        df_duplicated = st.session_state.df.duplicated().sum()
        st.write(f"There are {df_duplicated} duplicated rows.")

def fix():
    tab1, tab2, tab3 = st.tabs(
        ["Convert Data Types", "Handle Missing Values", "Drop Duplicated Rows"])

    with tab1:
     restore_backup()
     # Let user select a column
     column_to_convert = st.selectbox("Select a column to convert", st.session_state.df.columns)
     
     # Let user select the desired data type
     data_types = ["string", "datetime", "integer", "float"]
     new_type = st.selectbox("Select the new data type", data_types)
     
     # If datetime is selected, show format input
     if new_type == "datetime":
         date_format = st.text_input(
             "Enter the date format (e.g., '%m-%d-%Y', '%Y-%m-%d', or 'mixed' for automatic parsing)",
             value="mixed"
         )
     
     if st.button("Convert", key="convert_type_bt"):
         try:
             if new_type == "string":
                 st.session_state.df[column_to_convert] = st.session_state.df[column_to_convert].astype(str)
             elif new_type == "datetime":
                 if date_format.lower() == 'mixed':
                    st.session_state.df[column_to_convert] = pd.to_datetime(st.session_state.df[column_to_convert], errors='coerce')
                 else:
                    st.session_state.df[column_to_convert] = pd.to_datetime(st.session_state.df[column_to_convert], format=date_format)
             elif new_type == "integer":
                 st.session_state.df[column_to_convert] = st.session_state.df[column_to_convert].astype(int)
             elif new_type == "float":
                 st.session_state.df[column_to_convert] = st.session_state.df[column_to_convert].astype(float)
             
             st.success(f"Column '{column_to_convert}' converted to {new_type}")
             backup_df()        
        
         except Exception as e:
             st.error(f"Error converting column: {str(e)}")


    with tab2:
        restore_backup()
        # Get columns with missing values
        columns_with_missing = st.session_state.df.columns[st.session_state.df.isnull().any()].tolist()
        
        if not columns_with_missing:
            st.write("No columns with missing values found.")
        else:
            st.write("Select columns and choose how to handle missing values:")
            
            changes_made = False  # Flag to track if any changes were made
            actions = {}  # Dictionary to store actions for each column
            
            for column in columns_with_missing:
                st.markdown(f"<span style='background-color: gray;'>{column}</span>", unsafe_allow_html=True)
                action = st.selectbox(
                    f"Choose action for {column}",
                    ["Select an action", "Drop rows", "Drop column", "Fill with mean", 
                     "Fill with median", "Fill with mode", "Fill with constant"],
                    key=f"action_{column}"
                )
                
                if action != "Select an action":
                    changes_made = True
                    actions[column] = action
                
                if action == "Fill with constant":
                    constant_value = st.text_input(f"Enter constant value for {column}")
                    if constant_value:
                        actions[column] = (action, constant_value)
            
            # Only show the "Apply Changes" button if changes were made
            if changes_made:
                if st.button("Apply Changes", key="apply_changes_missing_bt"):
                    
                    for column, action in actions.items():
                        if action == "Drop rows":
                            st.session_state.df = st.session_state.df.dropna(subset=[column])
                            st.success(f"Rows with missing values in {column} have been dropped.")
                        
                        elif action == "Drop column":
                            st.session_state.df = st.session_state.df.drop(columns=[column])
                            st.success(f"Column {column} has been dropped.")
                        
                        elif action == "Fill with mean":
                            if pd.api.types.is_numeric_dtype(st.session_state.df[column]):
                                st.session_state.df[column].fillna(st.session_state.df[column].mean(), inplace=True)
                                st.success(f"Missing values in {column} have been filled with mean.")
                            else:
                                st.error(f"Cannot calculate mean for non-numeric column {column}.")
                        
                        elif action == "Fill with median":
                            if pd.api.types.is_numeric_dtype(st.session_state.df[column]):
                                st.session_state.df[column].fillna(st.session_state.df[column].median(), inplace=True)
                                st.success(f"Missing values in {column} have been filled with median.")
                            else:
                                st.error(f"Cannot calculate median for non-numeric column {column}.")
                        
                        elif action == "Fill with mode":
                            mode_value = st.session_state.df[column].mode().iloc[0]
                            st.session_state.df[column].fillna(mode_value, inplace=True)
                            st.success(f"Missing values in {column} have been filled with mode.")
                        
                        elif isinstance(action, tuple) and action[0] == "Fill with constant":
                            st.session_state.df[column].fillna(action[1], inplace=True)
                            st.success(f"Missing values in {column} have been filled with '{action[1]}'.")
                    
                    st.write("Dataset Updated!")
                    st.dataframe(st.session_state.df.head())
                    backup_df()


    with tab3:
        restore_backup()
        # Check for duplicate rows
        duplicate_rows = st.session_state.df.duplicated()
        num_duplicates = duplicate_rows.sum()
        
        if num_duplicates == 0:
            st.write("There are no duplicated rows in the dataset.")
        else:
            st.write(f"There are {num_duplicates} duplicate rows in the dataset.")
            
            if st.button("Drop Duplicates", key="drop_duplicates_bt"):
                st.session_state.df.drop_duplicates(inplace=True)
                st.success("All duplicate rows have been dropped.")
                backup_df()

def new_columns():
    restore_backup()

    tab1, tab2, tab3 = st.tabs(["Add New Columns", "Delete Columns", "Rename Columns"])

    with tab1:
        # Sidebar for managing functions
        with st.sidebar:
            st.header("Manage Custom Functions")
            
            # Add/Edit function
            st.subheader("Add/Edit Function")
            st.text_input("Function Name", key="function_name")
            st.text_area("Function Code", key="function_code")
            st.button("Save Function", on_click=add_or_update_function)
            
            # Remove function
            st.subheader("Remove Function")
            func_to_remove = st.selectbox("Select function to remove", 
                                          options=list(st.session_state.custom_functions.keys()),
                                          key="function_to_remove")
            st.button("Remove Function", on_click=remove_function)


        # Add a new column
        new_col_name = st.text_input("Enter new column name:")
        new_col_value = st.text_input("Enter value or formula (use df['column_name'] for existing columns):")

        if st.button("Add new column", key="add_new_column_bt"):
            if new_col_name and new_col_value:
                st.session_state.new_columns.append((new_col_name, new_col_value))
                st.success(f"Column '{new_col_name}' added to the queue. Click 'Apply Changes' to update the dataframe.")
            else:
                st.error("Please enter both column name and value/formula.")

        # Apply all new columns
        if st.button("Apply Changes", key="apply_changes_new_columns_bt"):
            success_columns = []
            error_columns = []
            
            # Create a local namespace with custom functions
            local_namespace = {
                'df': st.session_state.df,
                'pd': pd,
                'np': np
            }
            
            # Add custom functions to the local namespace
            for func_name, func_code in st.session_state.custom_functions.items():
                try:
                    exec(func_code, local_namespace)
                except Exception as e:
                    st.error(f"Error loading custom function '{func_name}': {str(e)}")
            
            for col_name, col_value in st.session_state.new_columns:
                try:
                    st.session_state.df[col_name] = eval(col_value, local_namespace)
                    success_columns.append(col_name)
                except Exception as e:
                    error_columns.append((col_name, str(e)))
            
            # Clear the queue for successfully added columns
            st.session_state.new_columns = [col for col in st.session_state.new_columns if col[0] not in success_columns]
            
            if success_columns:
                st.success(f"Successfully added columns: {', '.join(success_columns)}")
            if error_columns:
                for col_name, error in error_columns:
                    st.error(f"Error adding column '{col_name}': {error}")

            # Display the updated dataframe
        st.write("Updated DataFrame:")
        st.dataframe(st.session_state.df.head())
        backup_df()

    with tab2:

     restore_backup()   
     new_columns = [col for col in st.session_state.df.columns if col not in st.session_state.original_columns]
     if new_columns:
            
            columns_to_delete = st.multiselect("Select columns to delete:", new_columns, key="delete_columns")
            
            if st.button("Delete", key="delete_new_columns_bt"):
                if columns_to_delete:
                    st.session_state.df.drop(columns=columns_to_delete, inplace=True)
                    st.success(f"Deleted columns: {', '.join(columns_to_delete)}")
                    st.write(st.session_state.df.head())
                    # Reset the multiselect after deletion
                    st.session_state.columns_to_delete = []
                    backup_df()
                else:
                    st.warning("No columns selected for deletion.")
     else:
            st.info("No new columns available for deletion.")

    with tab3:
        # Rename columns
        # Get list of current column names
        current_columns = st.session_state.df.columns.tolist()
        
        # Select column to rename
        column_to_rename = st.selectbox("Select column to rename:", current_columns)
        
        # Enter new column name
        new_column_name = st.text_input("Enter new column name:", key="new_column_name")
        
        # Save button
        if st.button("Save", key="rename_column_bt"):
            if new_column_name and column_to_rename != new_column_name:
                try:
                    st.session_state.df = st.session_state.df.rename(columns={column_to_rename: new_column_name})
                    st.success(f"Column '{column_to_rename}' renamed to '{new_column_name}'")
                    backup_df()
                except Exception as e:
                    st.error(f"Error renaming column: {str(e)}")
            else:
                st.warning("Please enter a new column name different from the current name.")
        
        # Display the updated dataframe
        st.write("Updated DataFrame:")
        st.dataframe(st.session_state.df.head())

def export():
    restore_backup()

    # File uploader for importing settings
    json_file = st.file_uploader("Choose a settings file", type="json")
    if json_file is not None:
        st.session_state.json_file = json_file
        settings_json = json_file.getvalue().decode("utf-8")
        import_settings(settings_json)


    # Ensure the dataframe is not empty
    if 'df' in st.session_state and not st.session_state.df.empty:
        # Check if uploaded_file exists in session state and is valid
        uploaded_file = st.session_state.get('uploaded_file')
        if uploaded_file:
            # Generate the updated file name
            file_name = "updated_" + uploaded_file  # Adding _updated to the original filename
            file_name = file_name.replace(".csv", "_updated.csv")  # Ensure it has the .csv extension
        else:
            # Default filename if uploaded_file is None
            file_name = "updated_dataset.csv"

        # Convert DataFrame to CSV and offer it for download
        csv_data = st.session_state.df.to_csv(index=False)

        # Create a download button
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="Download Updated Dataset",
            data=csv_data,
            file_name=file_name,
            mime="text/csv"
        )
    else:
        st.error("No dataset available for export. Please upload or create a dataset first.")

    # Export settings functionality
    if st.button('Export Settings'):
        settings_json = export_settings()
        st.download_button(
            label="Download Settings",
            data=settings_json,
            file_name="app_settings.json",
            mime="application/json"
        )

def read_csv_with_encodings(uploaded_file):
    """Reads a CSV file trying multiple encodings."""
    encodings = [
        'utf-8', 'utf-8-sig', 'iso-8859-1', 'latin1', 'cp1252',
        'cp1251', 'utf-16', 'utf-16-le', 'utf-16-be'
    ]
    for encoding in encodings:
        try:
            stringio = StringIO(uploaded_file.getvalue().decode(encoding))
            return pd.read_csv(stringio), encoding
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return None, None


def upload_dataset():

    tab1, tab2, tab3 = st.tabs(['Single CSV', 'Join CSVs', 'to CSV'])

    with tab1:

        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            # List of possible encodings
            encodings = [
                'utf-8', 'utf-8-sig', 'iso-8859-1', 'latin1', 'cp1252',
                'cp1251', 'utf-16', 'utf-16-le', 'utf-16-be'
            ]
            
            df = None  # Initialize the dataframe
            successful_encoding = None  # Track the successful encoding
            
            # Try reading the file with each encoding
            for encoding in encodings:
                try:
                    # Decode and read the CSV
                    stringio = StringIO(uploaded_file.getvalue().decode(encoding))
                    df = pd.read_csv(stringio)
                    successful_encoding = encoding
                    break  # Exit loop if successful
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue  # Try the next encoding if this one fails
            
            if df is not None:
                st.success(f"File successfully decoded using '{successful_encoding}' encoding!")

            else:
                st.error("Failed to decode the file with the attempted encodings. Please check the file's format and encoding.")
        if uploaded_file is not None:
            try:
                # Check if a new file is uploaded
                if st.session_state.uploaded_file != uploaded_file.name:
                    # Clear charts when a new dataset is uploaded
                    st.session_state.charts = {}
                    st.session_state.layout = {}  # Optionally reset layout as well
                    st.success("Dashboard charts have been reset due to new dataset upload.")


                # Load the uploaded file into a DataFrame
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_file = uploaded_file.name  # Save the file name in session state
                st.session_state.original_columns = st.session_state.df.columns.tolist()
                backup_df()  # Save a backup of the dataset
                st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")
                st.write(st.session_state.df.head())  # Display the first few rows of the DataFrame
            except Exception as e:
                st.error(f"An error occurred while loading the dataset: {str(e)}")
        else:
            if st.session_state.df.empty:
                st.info("Please upload a dataset to proceed.")
            else:
                # Display existing dataset if already uploaded
                st.success(f"Using previously uploaded dataset: {st.session_state.uploaded_file}")
                st.write(st.session_state.df.head())
    with tab2:
        # File uploaders for two CSV files
        file1 = st.file_uploader("Upload first CSV file", type="csv")
        file2 = st.file_uploader("Upload second CSV file", type="csv")

        if file1 is not None and file2 is not None:
            # Read the first CSV file
            df1, encoding1 = read_csv_with_encodings(file1)
            if df1 is None:
                st.error("Failed to decode the first file with the attempted encodings.")
            else:
                st.success(f"First file successfully decoded using '{encoding1}' encoding!")
                st.dataframe(df1.head())

            # Read the second CSV file
            df2, encoding2 = read_csv_with_encodings(file2)
            if df2 is None:
                st.error("Failed to decode the second file with the attempted encodings.")
            else:
                st.success(f"Second file successfully decoded using '{encoding2}' encoding!")
                st.dataframe(df2.head())

            if df1 is not None and df2 is not None:
                # Allow user to select columns for joining
                join_column1 = st.selectbox("Select join column from first file", df1.columns)
                join_column2 = st.selectbox("Select join column from second file", df2.columns)
                
                # Allow user to select join type
                join_type = st.selectbox("Select join type", ["inner", "outer", "left", "right"])
                
                # Perform the join operation
                joined_df = pd.merge(df1, df2, left_on=join_column1, right_on=join_column2, how=join_type)
                
                # Display the joined DataFrame
                st.dataframe(joined_df.head(20))
                st.markdown(f"Rows: {joined_df.shape[0]:,}  \nColumns: {joined_df.shape[1]:,}")
                
                # Convert DataFrame to CSV
                csv = joined_df.to_csv(index=False)
                
                # Create download button
                st.download_button(
                    label="Download joined data as CSV",
                    data=csv,
                    file_name="joined_data.csv",
                    mime="text/csv",
                )
    with tab3:
        # Supported file formats
        supported_formats = ["xls", "xlsx", "xlt", "ods", "tsv", "sas7bdat", "sav", "mat", "rdata", "table"]

        # Streamlit file uploader
        uploaded_files = st.file_uploader(
            "Upload files in any supported format", 
            type=supported_formats, 
            accept_multiple_files=True
        )

        if st.button("Convert to CSV"):
            if uploaded_files:
                csv_files = []
                for file in uploaded_files:
                    try:
                        file_type = file.name.rsplit('.', 1)[-1].lower()
                        df = convert_to_dataframe(file, file_type)
                        csv_data = dataframe_to_csv(df)
                        csv_files.append((file.name.rsplit('.', 1)[0] + '.csv', csv_data))
                    except Exception as e:
                        st.error(f"Failed to process {file.name}: {e}")

                if csv_files:
                    # Create a zip file containing all CSV files
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for filename, data in csv_files:
                            zip_file.writestr(filename, data)

                    # Offer the zip file for download
                    st.download_button(
                        label="Download CSV files",
                        data=zip_buffer.getvalue(),
                        file_name="converted_files.zip",
                        mime="application/zip"
                    )
                else:
                    st.warning("No valid files to process.")
            else:
                st.warning("Please upload some files first.")



def add_filter():
    if 'filters' not in st.session_state or not isinstance(st.session_state.filters, list):
        st.session_state.filters = []
    st.session_state.filters = st.session_state.filters + [{}]

def remove_filter(index):
    filters = st.session_state.get('filters', [])
    filters.pop(index)
    st.session_state.filters = filters

def apply_filters(df):
    for f in st.session_state.get('filters', []):
        if f.get('column') and f.get('value') is not None:
            if pd.api.types.is_numeric_dtype(df[f['column']]):
                df = df[df[f['column']].between(*f['value'])]
            else:
                df = df[df[f['column']].isin(f['value'])]
    return df

def generate_chart_code(chart_type, df_name='df'):
    if chart_type in ["area_chart", "bar_chart", "line_chart", "scatter_chart"]:
        return f"""
        st.{chart_type}(
            data={df_name},
            x=None,
            y=None,
            x_label=None,
            y_label=None,
            color=None,
            width=None,
            height=None,
            use_container_width=True
        )
        """

    elif chart_type == "map":
        return f"""
        st.map(
            data={df_name},
            latitude='latitude_column',
            longitude='longitude_column',
            size='size_column',
            color='color_column',
            use_container_width=True
        )
        """
    elif chart_type == "pie_chart":
        return f"""
        import plotly.express as px

        fig = px.pie(
            {df_name}, 
            names='category_column', 
            values='value_column', 
            title='Pie Chart Example'
        )
        st.plotly_chart(fig, use_container_width=True)
        """
    elif chart_type == "histogram":
        return f"""
        import plotly.express as px

        fig = px.histogram(
            {df_name}, 
            x='x_column', 
            color='color_column', 
            title='Histogram Example'
        )
        st.plotly_chart(fig, use_container_width=True)
        """
    elif chart_type == "box_plot":
        return f"""
        import plotly.express as px

        fig = px.box(
            {df_name}, 
            x='category_column', 
            y='value_column', 
            color='category_column', 
            title='Box Plot Example'
        )
        st.plotly_chart(fig, use_container_width=True)
        """
    elif chart_type == "heatmap":
        return f"""
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        sns.heatmap({df_name}.corr(), annot=True, ax=ax)
        st.pyplot(fig)
        """
    elif chart_type == "violin_chart":
        return f"""
        import plotly.express as px

        fig = px.violin(
            {df_name}, 
            y='value_column', 
            x='category_column', 
            color='category_column', 
            box=True, 
            points='all', 
            title='Violin Plot Example'
        )
        st.plotly_chart(fig, use_container_width=True)
        """
    elif chart_type == "bubble_chart":
        return f"""
        import plotly.express as px

        fig = px.scatter(
            {df_name}, 
            x='x_column', 
            y='y_column', 
            size='size_column', 
            color='color_column', 
            title='Bubble Chart Example', 
            hover_name='hover_column'
        )
        st.plotly_chart(fig, use_container_width=True)
        """
    elif chart_type == "sunburst_chart":
        return f"""
        import plotly.express as px

        fig = px.sunburst(
            {df_name}, 
            path=['category_column_1', 'category_column_2'], 
            values='value_column', 
            title='Sunburst Chart Example'
        )
        st.plotly_chart(fig, use_container_width=True)
        """
    elif chart_type == "treemap":
        return f"""
        import plotly.express as px

        fig = px.treemap(
            {df_name}, 
            path=['category_column_1', 'category_column_2'], 
            values='value_column', 
            title='Treemap Example'
        )
        st.plotly_chart(fig, use_container_width=True)
        """
    elif chart_type == "streamgraph":
        return f"""
        import altair as alt

        streamgraph = alt.Chart({df_name}).mark_area().encode(
            x='x_column',
            y='y_column',
            color='category_column'
        )
        st.altair_chart(streamgraph, use_container_width=True)
        """
    elif chart_type == "candlestick_chart":
        return f"""
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Candlestick(
            x={df_name}['date_column'],
            open={df_name}['open_column'],
            high={df_name}['high_column'],
            low={df_name}['low_column'],
            close={df_name}['close_column']
        )])
        st.plotly_chart(fig, use_container_width=True)
        """
    elif chart_type == "radar_chart":
        return f"""
        import plotly.express as px

        fig = px.line_polar(
            {df_name}, 
            r='value_column', 
            theta='category_column', 
            color='group_column', 
            line_close=True,
            title='Radar Chart Example'
        )
        st.plotly_chart(fig, use_container_width=True)
        """
    elif chart_type == "wordcloud":
        return f"""
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        text = ' '.join({df_name}['text_column'])
        wordcloud = WordCloud(width=800, height=400).generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        """
    elif chart_type == "timeline_chart":
        return f"""
        import plotly.express as px

        fig = px.timeline(
            {df_name}, 
            x_start='start_column', 
            x_end='end_column', 
            y='category_column', 
            color='group_column',
            title='Timeline Chart Example'
        )
        st.plotly_chart(fig, use_container_width=True)
        """
    
    elif chart_type == "gauge_chart":
        return """
        import plotly.graph_objects as go

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=your_value,
            delta={{'reference': reference_value}},
            gauge={{'axis': {{'range': [None, max_value]}}}},
            title={{'text': "Gauge Chart Example"}}
        ))
        st.plotly_chart(fig, use_container_width=True)
        """
    elif chart_type == "kpi_card":
        return f"""
        st.metric(
            label="Metric Label",
            value="Current Value",
            delta="Delta Value",
            help="Additional context about the metric"
        )
        """
    elif chart_type == "density_chart":
        return f"""
        import plotly.express as px

        fig = px.density_contour(
            {df_name}, 
            x='x_column', 
            y='y_column', 
            color='category_column', 
            title='Density Chart Example'
        )
        st.plotly_chart(fig, use_container_width=True)
        """
    elif chart_type == "table":
        return f"""
        # Grouping and Aggregating Data with Lambda Functions
        grouped_df = {df_name}.groupby(['column1', 'column2'], as_index=False).agg(
            {{
                'column3': 'sum',  # Sum of column3
                'column4': lambda x: x.max() - x.min(),  # Custom aggregation: range
                'column5': 'mean'  # Mean of column5
            }}
        )

        # reset index
        grouped_df = grouped_df.reset_index()

        # Sorting by Multiple Columns
        sorted_df = grouped_df.sort_values(
            by=['column3', 'column4'],  # Sort by column3 and column4
            ascending=[False, True]  # Descending for column3, ascending for column4
        )

        # Selecting the Top Rows
        top_rows = sorted_df.head(10)

        # Display the Resulting DataFrame
        st.dataframe(top_rows)
        """
    else:
        return f"""
        # The chart type '{chart_type}' is not directly supported or requires additional customization.
        # Please add your custom code for '{chart_type}' below.

        # Example:
        # st.write("Custom implementation for {chart_type}")
        """

def report():
     tab1, tab2, tab3 = st.tabs(["Tables", "Filters", "Visualization"])
 
     with tab1: #Tables    
        restore_backup()
        
        # Let user select columns for grouping
        group_columns = st.multiselect("Select columns to group by:", st.session_state.df.columns)

        # Define available aggregation functions
        agg_functions = ["mean", "sum", "count", "min", "max", "median", "std", "var", "mode", "nunique", "quantile"]

        # Create a list to store aggregation selections
        if 'agg_list' not in st.session_state:
            st.session_state.agg_list = []

        # Let user add multiple aggregations
        col1, col2, col3 = st.columns(3)
        with col1:
            agg_column = st.selectbox("Select column:", st.session_state.df.columns)
        with col2:
            agg_function = st.selectbox("Select function:", agg_functions)
        with col3:
            if st.button("Add Aggregation", key="add_aggregation_tables"):
                st.session_state.agg_list.append((agg_column, agg_function))

        # Display and allow removal of selected aggregations
        for i, (col, func) in enumerate(st.session_state.agg_list):
            st.write(f"{i+1}. {col} - {func}")
            if st.button(f"Remove {i+1}", key=f"remove_aggregated{i+1}"):
                st.session_state.agg_list.pop(i)
                st.rerun()

        if group_columns and st.session_state.agg_list:
            # Create a dictionary for aggregation
            agg_dict = {f"{col}_{func}_{i}": (col, func) for i, (col, func) in enumerate(st.session_state.agg_list)}
            
            # Perform groupby and aggregation
            result = st.session_state.df.groupby(group_columns).agg(**agg_dict).reset_index()
            
            # Display the result
            st.write("Aggregated Report:")
            st.dataframe(result)

            # Save result functionality
            save_name = st.text_input("Enter a name to save this result TABLE:")
            if st.button("Save Result", key="save_results_tables"):
                if save_name:
                    if 'tb' not in st.session_state:
                        st.session_state.tb = {}
                    st.session_state.tb[save_name] = result
                    st.success(f"Result saved as '{save_name}'")
                else:
                    st.warning("Please enter a name to save the result.")

            # Display saved results
            if 'tb' in st.session_state and st.session_state.tb:
                st.write("Saved Results:")
                for name in st.session_state.tb.keys():
                    st.write(f"- {name}")
        else:
            st.write("Please select grouping columns and add at least one aggregation.")
     
     with tab2: #Filters
        restore_backup()
        # Step 1: Select dataframe
        dataframe_options = ["Original Dataframe"] + list(st.session_state.get('tb', {}).keys())
        selected_df_name = st.selectbox("Select a dataframe:", dataframe_options)
        
        if selected_df_name == "Original Dataframe":
            df = st.session_state.df
        else:
            df = st.session_state.tb[selected_df_name]
        
        # Step 2: Add filter conditions
        st.write("Add filter conditions:")
        if 'filter_conditions' not in st.session_state:
            st.session_state.filter_conditions = []
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            column = st.selectbox("Select column:", df.columns, key="filter_column")
        with col2:
            operator = st.selectbox("Select operator:", ["==", "!=", ">", "<", ">=", "<=", "contains"], key="filter_operator")
        with col3:
            value = st.text_input("Enter value:", key="filter_value")
        with col4:
            logic = st.selectbox("Logic:", ["AND", "OR"], key="filter_logic")
        with col5:
            if st.button("Add Condition", key="add_condition_filters"):
                st.session_state.filter_conditions.append((column, operator, value, logic))
                st.rerun()
        
        # Display and allow removal of filter conditions
        for i, (col, op, val, log) in enumerate(st.session_state.filter_conditions):
            st.write(f"{i+1}. {col} {op} {val} {log}")
            if st.button(f"Remove {i+1}", key=f"remove_condition{i+1}"):
                st.session_state.filter_conditions.pop(i)
                st.rerun()
        
        # Step 3: Apply filters
        if st.button("Apply Filters", key="apply_filters_bt"):
            filtered_df = df.copy()
            if st.session_state.filter_conditions:
                for i, (col, op, val, log) in enumerate(st.session_state.filter_conditions):
                    if op == "contains":
                        mask = filtered_df[col].astype(str).str.contains(val, case=False)
                    else:
                        mask = eval(f"filtered_df['{col}'] {op} {val}")
                    
                    if i == 0:  # First condition
                        filtered_df = filtered_df[mask]
                    elif log == "AND":
                        filtered_df = filtered_df[mask]
                    else:  # OR logic
                        filtered_df = pd.concat([filtered_df, df[mask]]).drop_duplicates()
            
            # Store the filtered dataframe in session state
            st.session_state.filtered_df = filtered_df
            st.write("Filtered Dataframe:")
            st.dataframe(st.session_state.filtered_df)

        # Option to save the filtered dataframe
        if 'filtered_df' in st.session_state:
            save_name = st.text_input("Enter a name to save this filtered result:")
            if st.button("Save Filtered Result", key="save_filters"):
                if save_name:
                    if 'tb' not in st.session_state:
                        st.session_state.tb = {}
                    st.session_state.tb[save_name] = st.session_state.filtered_df
                    st.success(f"Filtered result saved as '{save_name}'")
                else:
                    st.warning("Please enter a name to save the filtered result.")

        # Display saved results
        if 'tb' in st.session_state and st.session_state.tb:
            st.write("Saved Results:")
            for name in st.session_state.tb.keys():
                st.write(f"- {name}")

     with tab3: #Visualization
        import streamlit.components.v1 as components
        import pygwalker as pyg
        from pygwalker.api.streamlit import get_streamlit_html
        if df.empty:
            st.error("The DataFrame is empty. Please check your data source.")
        else:
            pyg_html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, spec_io_mode="json")

            components.html(pyg_html, height=1000)


def aggregate():

        # Let user select columns for grouping
        group_columns = st.multiselect("Select columns to group by:", st.session_state.df.columns)

        # Define available aggregation functions
        agg_functions = ["mean", "sum", "count", "min", "max", "median", "std", "var", "mode", "nunique", "quantile"]

        # Create a list to store aggregation selections
        # Ensure st.session_state.agg_list is a list of (column, function) pairs
        # Check if the list is not empty before accessing its first element
        if st.session_state.agg_list and not isinstance(st.session_state.agg_list[0], tuple):
            st.session_state.agg_list = [(col, func) for col, func in zip(st.session_state.agg_list[::2], st.session_state.agg_list[1::2])]
        
        # Let user add multiple aggregations
        col1, col2, col3 = st.columns(3)
        with col1:
            agg_column = st.selectbox("Select column:", st.session_state.df.columns)
        with col2:
            agg_function = st.selectbox("Select function:", agg_functions)
        with col3:
            if st.button("Add Aggregation", key="add_aggregation_tables"):
                st.session_state.agg_list.append((agg_column, agg_function))

        # Display and allow removal of selected aggregations
        for i, (col, func) in enumerate(st.session_state.agg_list):
            st.write(f"{i+1}. {col} - {func}")
            if st.button(f"Remove {i+1}", key=f"remove_aggregated{i+1}"):
                st.session_state.agg_list.pop(i)
                st.rerun()

        if group_columns and st.session_state.agg_list:
            # Create a dictionary for aggregation
            agg_dict = {}
            for col, func in st.session_state.agg_list:
                if func == 'quantile':
                    agg_dict[col] = lambda x: x.quantile(0.5)  # Using 0.5 for median, adjust as needed
                else:
                    agg_dict[col] = func
            
            # Perform groupby and aggregation
            result = st.session_state.df.groupby(group_columns).agg(agg_dict).reset_index()
            st.session_state.agg_code = f'st.session_state.df.groupby({group_columns}).agg({agg_dict}).reset_index()'
            
            # Display the result
            st.write("Aggregated Report:")
            st.dataframe(result)
            
            
            return st.session_state.agg_code





def dashboard_tab():
    
   
    
    tab1, tab2 = st.tabs(['TUI', 'GUI'])

    with tab1:#TUI

        
        restore_backup()
        # Check for saved results
        st.session_state.tb = st.session_state.get('tb', {})
        tb = pd.DataFrame(st.session_state.tb.items(), columns=['Key', 'Value'])
        st.session_state.custom_title = st.text_input("Enter dashboard title:", "Dashboard", key = "tui_title")

        with st.container(border = True):
            # Let the user define the dashboard layout
            rows = st.number_input("Number of rows", min_value=1, value=2)
            cols = st.number_input("Number of columns", min_value=1, value=2)

        # Create a list of cell positions
        cell_positions = [f"{i+1}-{j+1}" for i in range(rows) for j in range(cols)]

        # Store the layout in session state
        if "layout" not in st.session_state or st.session_state.layout != {"rows": rows, "cols": cols, "cells": cell_positions}:
            st.session_state.layout = {"rows": rows, "cols": cols, "cells": cell_positions}
            st.session_state.charts = {}  # Reset charts on layout change

        df = st.session_state.df
        st.dataframe(df.head(5))
        st.markdown(
            """
            <div style="
                background-color: green;
                color: white;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                margin-bottom: 20px;
            ">
           <strong>⚠️Warning:</strong> Before creating any chart, try to <B style="color: black;">aggregate</B>, <B style="color: black;">filter</B>, or create a <B style="color: black;">table</B> to ensure you are not displaying the whole DataFrame. Displaying the entire dataset will slow down the page and the app!
           </div>
            """,
            unsafe_allow_html=True
        )

        tb = False

        if tb:
            tb_keys = list(st.session_state.tb.keys())
            selected_key = st.selectbox("Select your created table", tb_keys)

            # Display the head of the selected key's data
            if selected_key:
                selected_data = st.session_state.tb[selected_key]
                if isinstance(selected_data, pd.DataFrame):
                    st.dataframe(selected_data.head())
        
        st.session_state.selected_df = df


        # List of available chart types
        chart_list = [
            "table",
            "area_chart", 
            "bar_chart", 
            "line_chart", 
            "scatter_chart", 
            "map", 
            "pie_chart", 
            "histogram", 
            "box_plot", 
            "heatmap", 
            "violin_chart", 
            "bubble_chart", 
            "sunburst_chart", 
            "treemap", 
            "streamgraph", 
            "candlestick_chart", 
            "radar_chart", 
            "wordcloud", 
            "timeline_chart", 
            "density_chart", 
            "gauge_chart", 
            "kpi_card"
            
        ]


        # Let the user select the chart type
        selected_chart = st.selectbox("Select a chart type", chart_list)

        col1, col2, col3 = st.columns([3,1,3], gap="small")
        with col1:
            # Let the user input their own code
            user_code = st.text_area("Enter your custom code for the chart:", value = f'agg_data = {st.session_state.agg_code}'+ "\n"+st.session_state.chart_code, height=200)
            

        with col2:
            column_types = pd.DataFrame({'Data Types': df.dtypes.astype(str)})
            st.dataframe(column_types, width=500)
        with col3:
            # Display sample code for the selected chart type
            if selected_chart:
                sample_code = generate_chart_code(selected_chart)
                st.code(sample_code, language='python')



    with tab2: #GUI
        
        restore_backup()
        # Check for saved results
        st.session_state.tb = st.session_state.get('tb', {})
        tb = pd.DataFrame(st.session_state.tb.items(), columns=['Key', 'Value'])
        st.session_state.custom_title = st.text_input("Enter dashboard title:", "Dashboard", key = "gui_title")

        with st.container(border = True):
            # Let the user define the dashboard layout
            rows = st.number_input("Number of rows", min_value=1, value=2, key='rows_gui')
            cols = st.number_input("Number of columns", min_value=1, value=2, key='cols_gui')

        # Create a list of cell positions
        cell_positions = [f"{i+1}-{j+1}" for i in range(rows) for j in range(cols)]

        # Store the layout in session state
        if "layout" not in st.session_state or st.session_state.layout != {"rows": rows, "cols": cols, "cells": cell_positions}:
            st.session_state.layout = {"rows": rows, "cols": cols, "cells": cell_positions}
            st.session_state.charts = {}  # Reset charts on layout change

        df = st.session_state.df
        st.dataframe(df.head(5))

        aggregate_choice = st.radio("Do you want to aggregate the dataset?", ("Yes", "No"))
        if aggregate_choice == "Yes":
            
            st.session_state.agg_code = aggregate()
            st.session_state.agg_event = True

        else:
            st.session_state.agg_event = False
            st.write("No aggregation performed.")




        # Chart Type Selection
        chart_options = [
                "None", "Area Chart", "Bar Chart", "Line Chart", "Scatter Chart",
                "Map", "Pie Chart", "Histogram", "Box Plot", "Heatmap",
                "Violin Chart", "Bubble Chart", "Sunburst Chart", "Treemap",
                "Streamgraph", "Candlestick Chart", "Radar Chart", "WordCloud",
                "Timeline Chart", "Density Chart", "Gauge Chart", "KPI Card"
            ]

        st.session_state["chart_type"] = st.selectbox(
            "Select the type of chart you want to create",
            chart_options,
            index=0 if st.session_state.get("chart_type") is None else chart_options.index(st.session_state["chart_type"]),
            placeholder="Choose a chart type..."
        )

        chart_type = st.session_state["chart_type"]


        # Dynamic Chart Creation
        if chart_type == "Area Chart":
            
            x = st.selectbox("X",df.columns)
            y = st.selectbox("Y", df.columns)
            color = st.selectbox("Color", [None]+list(df.columns) )
            x_label = st.text_input("X_label", value="", max_chars=None)
            y_label = st.text_input("Y_lable", value="", max_chars=None)
            width = st.number_input("Width", min_value=None, max_value=None, value=0, step=1)
            height = st.number_input("Height", min_value=None, max_value=None, value=0, step=1)
            use_container_width = st.checkbox("use container width", value=True)
            # user code
            user_code = textwrap.dedent(f'''st.area_chart(
                data = df,
                x = {repr(x)},
                y = {repr(y)},
                color = {repr(color)},
                x_label = {repr(x_label)},
                y_label = {repr(y_label)},
                width = {width},
                height = {height},
                use_container_width = {use_container_width}
             )

            ''')


        if chart_type == "Bar Chart":

            x = st.selectbox("X",df.columns)
            y = st.selectbox("Y", df.columns)
            color = st.selectbox("Color", [None]+list(df.columns) )
            x_label = st.text_input("X_label", value="", max_chars=None)
            y_label = st.text_input("Y_lable", value="", max_chars=None)
            width = st.number_input("Width", min_value=None, max_value=None, value=0, step=1)
            height = st.number_input("Height", min_value=None, max_value=None, value=0, step=1)
            use_container_width = st.checkbox("use container width", value=True)

            if st.session_state.agg_event:
                data = 'agg_data'
            else:
                data = 'df'

            st.session_state.chart_code = textwrap.dedent(f'''st.bar_chart(
                data = {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
                x = {repr(x)},
                y = {repr(y)},
                color = {repr(color)},
                x_label = {repr(x_label)},
                y_label = {repr(y_label)},
                width = {width},
                height = {height},
                use_container_width = {use_container_width}
             )''')

            

        if chart_type == "Line Chart":
            x = st.selectbox("X",df.columns)
            y = st.selectbox("Y", df.columns)
            color = st.selectbox("Color", [None]+list(df.columns) )
            x_label = st.text_input("X_label", value="", max_chars=None)
            y_label = st.text_input("Y_lable", value="", max_chars=None)
            width = st.number_input("Width", min_value=None, max_value=None, value=0, step=1)
            height = st.number_input("Height", min_value=None, max_value=None, value=0, step=1)
            use_container_width = st.checkbox("use container width", value=True)
            # user code
            user_code = textwrap.dedent(f"""st.line_chart(
                data=df,
                x={repr(x)},
                y={repr(y)},
                color={repr(color)},
                x_label={repr(x_label)},
                y_label={repr(y_label)},
                width={width},
                height={height},
                use_container_width={use_container_width}
            )""")


        if chart_type == "Scatter Chart":
            x = st.selectbox("X",df.columns)
            y = st.selectbox("Y", df.columns)
            color = st.selectbox("Color", [None]+list(df.columns) )
            x_label = st.text_input("X_label", value="", max_chars=None)
            y_label = st.text_input("Y_lable", value="", max_chars=None)
            width = st.number_input("Width", min_value=None, max_value=None, value=0, step=1)
            height = st.number_input("Height", min_value=None, max_value=None, value=0, step=1)
            use_container_width = st.checkbox("use container width", value=True)
            # user code
            user_code = textwrap.dedent(f'''st.scatter_chart(
                data = df,
                x = {repr(x)},
                y = {repr(y)},
                color = {repr(color)},
                x_label = {repr(x_label)},
                y_label = {repr(y_label)},
                width = {width},
                height = {height},
                use_container_width = {use_container_width}
             )

            ''')
        

        if chart_type == "Map":
            latitude = st.selectbox("Latitude", df.columns)
            longitude = st.selectbox("Longitude", df.columns)
            size = st.selectbox("Size",[None]+list(df.columns))
            color = st.selectbox("Color", [None]+list(df.columns))
            use_container_width = st.checkbox("use container width", value=True)
            # user code
            user_code = textwrap.dedent(f"""st.map(
                data = df,
                latitude = {repr(latitude)},
                longitude = {repr(longitude)},
                size = {repr(size)},
                color = {repr(color)},
                use_container_width = {use_container_width}

            )
                        """)


        if chart_type == "Pie Chart":
            names = st.selectbox("Category Column", df.columns)
            values = st.selectbox("Value Column", df.columns)
            use_container_width = st.checkbox("use container width", value=True)
            # user code
            user_code = textwrap.dedent(f"""
                fig = px.pie(
                df,
                names = {repr(names)},
                values = {repr(values)},
                )
                st.plotly_chart(fig)
            """)

        
        if chart_type == "Histogram":
            x = st.selectbox("X", df.columns)
            color = st.selectbox("Color", [None]+list(df.columns))
            use_container_width = st.checkbox("use container width", value=True)

            user_code = textwrap.dedent(f"""
                fig = px.histogram(
                df,
                x = {repr(x)},
                color = {repr(color)},
                )
                st.plotly_chart(fig,use_container_width = {use_container_width} )
                """
                )

        if chart_type == "Box Plot":
            x = st.selectbox("Category Column", df.columns)
            y = st.selectbox("Value Column", df.columns)
            color = st.selectbox("Color", [None]+list(df.columns))
            use_container_width = st.checkbox("use container width", value=True)

            user_code = textwrap.dedent(f"""
                fig = px.box(
                    df,
                    x = {repr(x)},
                    y = {repr(y)},
                    color = {repr(color)}
                    )
                st.plotly_chart(fig, {use_container_width})
                """
                )

        if chart_type == "Heatmap":

            selected_columns = st.multiselect("Columns", df.columns.tolist(), default=df.columns.tolist())
            corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
            color_palette = st.selectbox("Color Palette", ["coolwarm", "viridis", "YlGnBu", "RdYlBu"])
            show_annotations = st.checkbox("Show correlation values", value=True)
            font_size = st.slider("Annotation font size", 6, 20, 10)

            user_code = textwrap.dedent(f"""
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = df[{repr(selected_columns)}].corr(method={repr(corr_method)})
                sns.heatmap(
                    correlation_matrix, 
                    annot={repr(show_annotations)}, 
                    fmt=".2f", 
                    cmap={repr(color_palette)}, 
                    ax=ax, 
                    annot_kws={{"size": {font_size}}}
                )
                st.pyplot(fig)
                """)

        if chart_type == "Violin Chart":

            # Column selection
            value_column = st.selectbox("Value Column (y-axis)", df.columns.tolist())
            category_column = st.selectbox("Category Column (x-axis and color)", df.columns.tolist())

            # Plot customization
            show_box = st.checkbox("Show box plot inside violin", value=True)
            show_points = st.checkbox("Show all data points", value=True)

            user_code = textwrap.dedent(f"""
                fig = px.violin(
                    df,
                    y={repr(value_column)},
                    x={repr(category_column)},
                    color={repr(category_column)},
                    box={repr(show_box)},
                    points={'"all"' if show_points else False},
                    
                )
                st.plotly_chart(fig, use_container_width=True)
                """)

        if chart_type == "Bubble Chart":

            # Column selection
            x_column = st.selectbox("x-axis column", df.columns.tolist())
            y_column = st.selectbox("y-axis column", df.columns.tolist())
            size_column = st.selectbox("bubble size column", df.columns.tolist())
            color_column = st.selectbox("color column", df.columns.tolist())
            hover_column = st.selectbox("hover data column", df.columns.tolist())

            user_code = textwrap.dedent(f"""
                fig = px.scatter(
                    df,
                    x={repr(x_column)},
                    y={repr(y_column)},
                    size={repr(size_column)},
                    color={repr(color_column)},
                    hover_name={repr(hover_column)}
                )
                st.plotly_chart(fig, use_container_width=True)
                """)

        if chart_type == "Sunburst Chart":
            # Column selection
            all_columns = df.columns.tolist()
            path_columns = st.multiselect("Select hierarchical category columns (in order)", all_columns, max_selections=3)
            value_column = st.selectbox("Select the value column", all_columns)

            user_code = textwrap.dedent(f"""
            if {repr(path_columns)} and {repr(value_column)}:
                fig = px.sunburst(
                    df,
                    path={repr(path_columns)},
                    values={repr(value_column)},
                    
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Please select at least one path column and a value column.")
            """)


        if chart_type == "Treemap":
            # Column selection
            all_columns = df.columns.tolist()
            path_columns = st.multiselect("Select hierarchical category columns (in order)", all_columns, max_selections=3)
            value_column = st.selectbox("Select the value column", all_columns)

            user_code = textwrap.dedent(f"""
                if {repr(path_columns)} and {repr(value_column)}:
                    fig = px.treemap(
                        df,
                        path={repr(path_columns)},
                        values={repr(value_column)},

                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Please select at least one path column and a value column.")
                """)


        if chart_type == "Streamgraph":
            # Column selection
            x_column = st.selectbox("Select the x-axis column (e.g., time)", df.columns.tolist())
            y_column = st.selectbox("Select the y-axis column (numerical data)", df.columns.tolist())
            category_column = st.selectbox("Select the category column", df.columns.tolist())

            # Plot customization
            color_scheme = st.selectbox("Select color scheme", ["category10", "category20", "tableau10", "tableau20"])

            user_code = textwrap.dedent(f"""
                streamgraph = alt.Chart(df).mark_area().encode(
                    x={repr(x_column)},
                    y=alt.Y({repr(y_column)}, stack='center'),
                    color=alt.Color({repr(category_column)}, scale=alt.Scale(scheme={repr(color_scheme)}))
                ).properties(

                )
                st.altair_chart(streamgraph, use_container_width=True)
                """)


        if chart_type == "Candlestick Chart":
            # Column selection
            date_column = st.selectbox("Select the date column", [""] + df.columns.tolist())
            open_column = st.selectbox("Select the open price column", [""] + df.columns.tolist())
            high_column = st.selectbox("Select the high price column", [""] + df.columns.tolist())
            low_column = st.selectbox("Select the low price column", [""] + df.columns.tolist())
            close_column = st.selectbox("Select the close price column", [""] + df.columns.tolist())

            # Initialize date_range with a default value
            date_range = None

            # Optional date range selection
            if date_column:
                try:
                    min_date = df[date_column].min()
                    max_date = df[date_column].max()
                    date_range = st.date_input("Select date range", [min_date, max_date])
                except KeyError:
                    st.error(f"The selected date column '{date_column}' is not valid. Please choose a valid date column.")
            else:
                st.warning("Please select a date column before setting the date range.")

            # Use date_range only if it's defined and date_column is selected
            if date_range and date_column:
                df_filtered = df[(df[date_column] >= str(date_range[0])) & (df[date_column] <= str(date_range[1]))]
            else:
                df_filtered = df  # Use the original dataframe if date_range is not set


            if date_range and date_column and all([open_column, high_column, low_column, close_column]):

                user_code = textwrap.dedent(f"""
                    df_filtered = df[(df[{repr(date_column)}] >= str({repr(date_range[0])})) & (df[{repr(date_column)}] <= str({repr(date_range[1])}))]

                    fig = go.Figure(data=[go.Candlestick(
                        x=df_filtered[{repr(date_column)}],
                        open=df_filtered[{repr(open_column)}],
                        high=df_filtered[{repr(high_column)}],
                        low=df_filtered[{repr(low_column)}],
                        close=df_filtered[{repr(close_column)}]
                    )])

                    st.plotly_chart(fig, use_container_width=True)
                    """)


        if chart_type == "Radar Chart":
            # Column selection
            value_column = st.selectbox("Select the value column (radial axis)", df.columns.tolist())
            category_column = st.selectbox("Select the category column (angular axis)", df.columns.tolist())
            group_column = st.selectbox("Select the group column (for multiple traces)", df.columns.tolist())

            # Plot customization
            color_scheme = st.selectbox("Select color scheme", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"])

            user_code = textwrap.dedent(f"""
                fig = px.line_polar(
                    df,
                    r={repr(value_column)},
                    theta={repr(category_column)},
                    color={repr(group_column)},
                    line_close=True,
                    title="Polar Line Chart Example",
                    color_discrete_sequence=getattr(px.colors.sequential, {repr(color_scheme)})
                )
                st.plotly_chart(fig, use_container_width=True)
                """)


        if chart_type == "WordCloud":
            # Column selection
            text_column = st.selectbox("Select the text column for word cloud", df.columns.tolist())

            # WordCloud customization
            width = st.slider("Word cloud width", 400, 1200, 800)
            height = st.slider("Word cloud height", 200, 800, 400)
            bg_color = st.color_picker("Background color", "#FFFFFF")
            max_words = st.slider("Maximum words", 50, 500, 200)
            colormap = st.selectbox("Color scheme", ["viridis", "plasma", "inferno", "magma"])

            user_code = textwrap.dedent(f"""
            text = ' '.join(df[{repr(text_column)}].dropna().astype(str))
            wordcloud = WordCloud(
                width={repr(width)},
                height={repr(height)},
                background_color={repr(bg_color)},
                max_words={repr(max_words)},
                colormap={repr(colormap)}
            ).generate(text)

            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            """)


        if chart_type == "Timeline Chart":
            # Column selection
            start_column = st.selectbox("Select the start date/time column", df.columns.tolist())
            end_column = st.selectbox("Select the end date/time column", df.columns.tolist())
            category_column = st.selectbox("Select the category column (y-axis)", df.columns.tolist())
            group_column = st.selectbox("Select the group column (for color-coding)", df.columns.tolist())

            # Plot customization
            color_scheme = st.selectbox("Select color scheme", ["Plotly", "D3", "G10", "T10", "Alphabet"])

            user_code = textwrap.dedent(f"""
                fig = px.timeline(
                    df,
                    x_start={repr(start_column)},
                    x_end={repr(end_column)},
                    y={repr(category_column)},
                    color={repr(group_column)},
                    color_discrete_sequence=getattr(px.colors.qualitative, {repr(color_scheme)})
                )
                st.plotly_chart(fig, use_container_width=True)
            """)


        if chart_type == "Density Chart":
            # Column selection
            x_column = st.selectbox("Select the x-axis column", df.columns.tolist())
            y_column = st.selectbox("Select the y-axis column", df.columns.tolist())
            category_column = st.selectbox("Select the category column (for color-coding)", df.columns.tolist())

            # Plot customization
            color_scheme = st.selectbox("Select color scheme", ["Plotly", "D3", "G10", "T10", "Alphabet"])

            user_code = textwrap.dedent(f"""
                fig = px.density_contour(
                    df,
                    x={repr(x_column)},
                    y={repr(y_column)},
                    color={repr(category_column)},
                    title="Density Contour Plot Example",
                    color_discrete_sequence=getattr(px.colors.qualitative, {repr(color_scheme)})
                )
                st.plotly_chart(fig, use_container_width=True)
                """)


        if chart_type == "Gauge Chart":
            # Column selection
            value_column = st.selectbox("Select the column for gauge value", df.columns.tolist())
            reference_column = st.selectbox("Select the column for reference value (optional)", ["None"] + df.columns.tolist())
            
            # Gauge customization
              # or let the user input it
            if value_column:

                max_value = st.number_input("Enter the maximum value for the gauge", value=float(df[value_column].max()))

                user_code = textwrap.dedent(f"""
                    gauge_value = df[{repr(value_column)}].iloc[-1]

                    reference_value = None
                    if {repr(reference_column)} != "None":
                        reference_value = df[{repr(reference_column)}].iloc[-1]

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta" if reference_value else "gauge+number",
                        value=gauge_value,
                        delta={{'reference': reference_value}} if reference_value else None,
                        gauge={{'axis': {{'range': [None, {max_value}]}}}},
                        
                    ))

                    st.plotly_chart(fig, use_container_width=True)
                    """)


        if chart_type == "KPI Card":

            # Get list of numeric columns for value and delta
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Get list of all columns for label
            all_columns = df.columns.tolist()

            # User inputs
            label_column = st.selectbox("KPI label:", all_columns)
            value_column = st.selectbox("current KPI value:", numeric_columns)
            delta_column = st.selectbox("KPI change (delta):", ["None"] + numeric_columns)
            
            # Optional configurations
            delta_color = st.selectbox("delta color:", ["normal", "inverse", "off"])
            label_visibility = st.selectbox("label visibility?", ["visible", "hidden", "collapsed"])
            border = st.checkbox("border around the metric?")

            user_code = textwrap.dedent(f"""
                # Get the latest values from the selected columns
                label = df[{repr(label_column)}].iloc[-1]
                value = df[{repr(value_column)}].iloc[-1]
                delta = df[{repr(delta_column)}].iloc[-1] if {repr(delta_column)} != "None" else None

                st.metric(
                    label=label,
                    value=value,
                    delta=delta,
                    delta_color={repr(delta_color)},
                    label_visibility={repr(label_visibility)},
                    border={repr(border)}
                )
                """)






    # Ask for chart title and axis labels
    chart_title = st.text_input("Chart title")

    # Let the user select the cell position
    selected_cell = st.selectbox("Select cell position", st.session_state.layout["cells"])

    # Create the chart when the user clicks a button
    if st.button("Create Chart"):
        st.session_state.charts[selected_cell] = {
            "type": "custom",
            "code": user_code,
            "title": chart_title,
            "data": df  # Store the actual dataframe snapshot
        }
        st.success(f"Chart created and placed in cell {selected_cell}")



    # Display the dashboard

    dashboard()




def dashboard():


    with st.sidebar:
        df = st.session_state.selected_df
        st.write("Dashboard Filters")
         
        # Button to add a new filter
        st.button("Add Filter", on_click=add_filter, key="dashboard_filters")

        # Display and configure each filter
        for i, filter in enumerate(st.session_state.get('filters', [])):
            with st.expander(f"Filter {i+1}", expanded=True):
                col1, col2, col3 = st.columns([2,2,1])
                
                with col1:
                    filter['column'] = st.selectbox("Select Column", df.columns, key=f"col_{i}")
                
                with col2:
                    if filter['column']:
                        if pd.api.types.is_numeric_dtype(df[filter['column']]):
                            min_val, max_val = float(df[filter['column']].min()), float(df[filter['column']].max())
                            
                            # Use separate number inputs without columns
                            start_value = st.number_input(f"Start value for {filter['column']}", 
                                                          value=min_val, 
                                                          min_value=min_val, 
                                                          max_value=max_val, 
                                                          key=f"start_{i}")
                            
                            end_value = st.number_input(f"End value for {filter['column']}", 
                                                        value=max_val, 
                                                        min_value=min_val, 
                                                        max_value=max_val, 
                                                        key=f"end_{i}")
                            
                            # Use the input values for the slider
                            filter['value'] = st.slider("Select Range", 
                                                        min_value=min_val, 
                                                        max_value=max_val, 
                                                        value=(start_value, end_value), 
                                                        key=f"val_{i}")
                        else:
                            options = df[filter['column']].unique().tolist()
                            
                            # Use multiselect for both search and selection
                            selected_values = st.multiselect(
                                f"Select values for {filter['column']}",
                                options=options,
                                default=options if st.checkbox("Select All", key=f"select_all_{i}") else [],
                                key=f"multiselect_{i}"
                            )
                            
                            filter['value'] = selected_values

                            # Show how many options are selected
                            st.write(f"{len(selected_values)} option(s) selected.")
                with col3:
                    st.button("Remove", key=f"remove_{i}", on_click=remove_filter, args=(i,))
        if st.button("Update Dashboard", key = "update"):
            df = apply_filters(df)   



 


    
    st.write(df.head())

    if "rows" not in st.session_state.layout or "cols" not in st.session_state.layout:
        st.error("Dashboard layout is not configured. Please set it up first.")
        return

    
    
    
    st.session_state.custom_dashboard_title = st.session_state.custom_title

    
    
    custom_title = st.session_state.get("custom_dashboard_title", "Dashboard")
    
    # Display the dashboard with custom title
    # Center and display the title
    st.markdown(
        f"""
        <h1 style="text-align: center; margin-top: 0px;">
            {custom_title}
        </h1>
        """,
        unsafe_allow_html=True
    )

    

    for i in range(st.session_state.layout["rows"]):
        cols = st.columns(st.session_state.layout["cols"])
        for j, col in enumerate(cols):
            cell = f"{i+1}-{j+1}"
            if cell in st.session_state.charts:
                chart_data = st.session_state.charts[cell]
                with col:
                    st.subheader(chart_data["title"])
                    try:
                        
                        # Execute the custom code with the saved dataframe
                        exec(chart_data["code"], {"df": df, "tb": st.session_state.tb, "st": st, "px":px, "plt": plt, "sns":sns, "alt": alt, "datetime": datetime, "go":go, "WordCloud": WordCloud})
                    except Exception as e:
                        st.error(f"Error executing custom code: {str(e)}")
                        st.error(f"Chart data: {chart_data}")

    


def export_settings():
    # Convert the DataFrame to a dictionary only if it's not empty
    df_dict = st.session_state.df.to_dict(orient='split') if not st.session_state.df.empty else {}

    # Convert charts to a serializable format
    charts_serializable = {}
    for cell, chart in st.session_state.get('charts', {}).items():
        # Check if 'data' is a DataFrame and convert it
        chart_data = chart.get('data', None)
        if isinstance(chart_data, pd.DataFrame):
            chart_data = chart_data.fillna('')  # Fill missing values if necessary
            chart_data = chart_data.to_dict(orient='split')  # Convert DataFrame to dictionary

        # Update the chart data with the serializable format
        charts_serializable[cell] = {
            'type': chart['type'],
            'code': chart['code'],
            'title': chart['title'],
            'data': chart_data,
        }

    # Get other session state values
    new_columns = st.session_state.get('new_columns', [])
    layout = st.session_state.get('layout', {})

    # Create the settings dictionary
    settings = {
        'df': df_dict,
        'new_columns': new_columns,
        'layout': layout,
        'charts': charts_serializable,  # Use the serialized charts
        'filters': st.session_state.get('filters', []),
    }

    # Custom JSON encoder to handle Timestamps
    def custom_json_encoder(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()  # Convert to ISO 8601 string
        raise TypeError(f"Type {type(obj)} not serializable")

    # Return the serialized JSON string with the custom encoder
    return json.dumps(settings, default=custom_json_encoder)


def import_settings(settings_json):
    try:
        # Load settings from the JSON data
        settings = json.loads(settings_json)

        # Rebuild the DataFrame from the 'df' part of the settings if it's not empty
        df_dict = settings.get('df', {})
        if df_dict and 'data' in df_dict and 'columns' in df_dict:
            # Rebuild DataFrame using the unpacking operator for 'split' orientation
            st.session_state.df = pd.DataFrame(data=df_dict['data'], columns=df_dict['columns'])
            if 'index' in df_dict:
                st.session_state.df.index = pd.Index(df_dict['index'])
        else:
            st.session_state.df = pd.DataFrame()  # Empty DataFrame if 'df' is not in settings
        
        # Rebuild other session state values
        st.session_state.new_columns = settings.get('new_columns', [])
        st.session_state.layout = settings.get('layout', {})
        
        # Rebuild charts with data conversion (if any data is a DataFrame)
        charts_data = settings.get('charts', {})
        for cell, chart in charts_data.items():
            # Convert 'data' back into a DataFrame if it's a dictionary representation of a DataFrame
            chart_data = chart.get('data', None)
            if chart_data and 'data' in chart_data and 'columns' in chart_data:
                chart_data = pd.DataFrame(data=chart_data['data'], columns=chart_data['columns'])
                if 'index' in chart_data:
                    chart_data.index = pd.Index(chart_data['index'])

            # Store the chart back into session state
            st.session_state.charts[cell] = {
                'type': chart['type'],
                'code': chart['code'],
                'title': chart['title'],
                'data': chart_data,
            }

        # Handle any other session state variables (filters, etc.)
        st.session_state.filters = settings.get('filters', [])

        st.success("Settings imported successfully!")
    except Exception as e:
        st.error(f"Error importing settings: {e}")


def convert_to_dataframe(file, file_type):
    """
    Convert the uploaded file to a pandas DataFrame based on its file type.
    """
    if file_type in ["xls", "xlsx", "xlt"]:
        df = pd.read_excel(file)
    elif file_type == "ods":
        df = pd.read_excel(file, engine="odf")
    elif file_type == "tsv":
        df = pd.read_csv(file, sep="\t")
    elif file_type == "sas7bdat":
        import sas7bdat
        with sas7bdat.SAS7BDAT(file) as f:
            df = f.to_data_frame()
    elif file_type == "sav":
        import pyreadstat
        df, meta = pyreadstat.read_sav(file)
    elif file_type == "mat":
        from scipy.io import loadmat
        mat = loadmat(file)
        df = pd.DataFrame({key: mat[key].flatten() for key in mat if not key.startswith("__")})
    elif file_type == "rdata":
        import pyreadr
        result = pyreadr.read_r(file)
        df = next(iter(result.values()))  # Extract the first DataFrame
    elif file_type == "table":
        df = pd.read_table(file)
    else:
        raise ValueError("Unsupported file format")
    return df

def dataframe_to_csv(dataframe):
    """
    Convert a DataFrame to CSV format and return as bytes.
    """
    csv_buffer = BytesIO()
    dataframe.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


# Show the dashboard if toggle is active
# Show the dashboard if toggle is active
if show_only_dashboard:
    dashboard()
else:
    # header
    st.markdown("<h2 style='text-align: center;'>Welcome to DataPhil!👋</h2>",
            unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>Your Data, Your Way—Fast & Easy!💩</h6>", 
            unsafe_allow_html=True)

    # sections
    section_selection = st.pills("", ["Upload Dataset", "Summary", "Fix Dataset", "New Columns", "Settings", "Report", "Dashboard"])
    # Display content based on sidebar selection
    if section_selection == "Upload Dataset":
        upload_dataset()

    elif section_selection == "Summary":
        summary()

    elif section_selection == "Fix Dataset":
        fix()

    elif section_selection == "New Columns":
        new_columns()

    elif section_selection == "Settings":
        export()

    elif section_selection == "Report":
        report()
    elif section_selection == "Dashboard":
        dashboard_tab()