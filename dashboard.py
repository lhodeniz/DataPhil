# import modules

import pandas as pd
import streamlit as st
import numpy as np
import os
import json
import io
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
import boto3
import uuid
import hashlib
import time
import re
from streamlit_ace import st_ace


##########    page config   ############

st.set_page_config(page_title="DataPhil", layout="wide", initial_sidebar_state="collapsed")


show_only_dashboard = st.toggle("Show Only Dashboard")


############ initializations #####################

if "section_selection" not in st.session_state:
    st.session_state.section_selection = "Upload Dataset"

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

if 'selected_df' not in st.session_state:
      st.session_state.selected_df = pd.DataFrame()

if 'agg_result' not in st.session_state:
      st.session_state.agg_result = pd.DataFrame()

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

if 'column_widths' not in st.session_state:
    column_widths = []

if 'df_name' not in st.session_state:
    st.session_state.df_name = ''

############# CSS ####################

with open("css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


###########    FUNCTIONS   #################

def add_or_update_function():
    func_name = st.session_state.function_name
    func_code = st.session_state.function_code
    if func_name and func_code:
        st.session_state.custom_functions[func_name] = func_code
        st.success(f"Function '{func_name}' added/updated successfully!")


def remove_function():
    func_name = st.session_state.function_to_remove
    if func_name in st.session_state.custom_functions:
        del st.session_state.custom_functions[func_name]
        st.success(f"Function '{func_name}' removed successfully!")

def calculate_df_hash(df):
    """Calculate a hash for the given DataFrame."""
    df_string = df.to_json()  # Convert to a JSON string
    return hashlib.md5(df_string.encode()).hexdigest()

def calculate_dashboard_hash(layout, charts):
    """Calculate a hash for the dashboard layout and charts."""
    def make_serializable(obj):
        """Helper function to convert non-serializable objects to strings."""
        if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
            return obj
        return str(obj)  # Convert other types to strings

    # Preprocess the layout and charts to ensure serializability
    serializable_layout = json.loads(json.dumps(layout, default=make_serializable))
    serializable_charts = json.loads(json.dumps(charts, default=make_serializable))
    
    dashboard_state = {
        "layout": serializable_layout,
        "charts": serializable_charts
    }
    dashboard_json = json.dumps(dashboard_state, sort_keys=True)
    return hashlib.md5(dashboard_json.encode()).hexdigest()


def backup_df():
    if 'df_backup' not in st.session_state or not st.session_state.df_backup.equals(st.session_state.df):
        st.session_state.df_backup = st.session_state.df.copy()

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


def adjust_dashboard_layout():
    """Function to adjust column widths for the dashboard layout."""
    # Retrieve current layout dimensions
    num_rows = st.session_state.layout.get("rows", 2)
    num_cols = st.session_state.layout.get("cols", 2)

    layout_widths = []
    with st.container(key = "dashboard_adjustments"):
        #
        for row_index in range(num_rows):
            st.write(f"Row {row_index + 1} column widths:")
            row_widths = []
            cols = st.columns(num_cols)  # Create a row of columns
            for col_index, col in enumerate(cols):
                with col:
                    col_width = st.slider(
                        f"Col {col_index + 1}",
                        min_value=0, max_value=100, value=50, step=5,
                        key=f"slider_{row_index}_{col_index}"
                    )
                row_widths.append(col_width)
            layout_widths.append(row_widths)

    # Save the column widths in session state
    st.session_state.column_widths = layout_widths
  

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



############ MAIN FUNCTIONS ################

def upload_dataset():

    tab1, tab2, tab3 = st.tabs(['Single CSV', 'Join CSVs', 'to CSV'])

    with tab1:
        #
        @st.cache_data
        def load_csv(file, encoding):
            stringio = StringIO(file.getvalue().decode(encoding))
            return pd.read_csv(stringio)

        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            # List of possible encodings
            encodings = [
                'utf-8', 'utf-8-sig', 'iso-8859-1', 'latin1', 'cp1252',
                'cp1251', 'utf-16', 'utf-16-le', 'utf-16-be'
            ]
            
            df = None
            successful_encoding = None
            
            # Try reading the file with each encoding
            for encoding in encodings:
                try:
                    df = load_csv(uploaded_file, encoding)
                    successful_encoding = encoding
                    break  # Exit loop if successful
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue  # Try the next encoding if this one fails
            
            if df is not None:
                st.success(f"File successfully decoded using '{successful_encoding}' encoding!")
                
                # Check if a new file is uploaded
                if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file != uploaded_file.name:
                    # Clear charts when a new dataset is uploaded
                    st.session_state.charts = {}
                    st.session_state.layout = {}  # Optionally reset layout as well
                    st.success("Dashboard charts have been reset due to new dataset upload.")
                
                st.session_state.df = df
                st.session_state.uploaded_file = uploaded_file.name
                st.session_state.original_columns = df.columns.tolist()
                backup_df()  # Save a backup of the dataset
                st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")
                st.write(df.head())  # Display the first few rows of the DataFrame
            else:
                st.error("Failed to decode the file with the attempted encodings. Please check the file's format and encoding.")
        else:
            if 'df' in st.session_state and not st.session_state.df.empty:
                # Display existing dataset if already uploaded
                st.success(f"Using previously uploaded dataset: {st.session_state.uploaded_file}")
                st.write(st.session_state.df.head())
            else:
                st.info("Please upload a dataset to proceed.")

        if st.session_state.uploaded_file:

            df_name = uploaded_file.name
            st.session_state.df_name = df_name


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


def summary():
    # Check if the dataframe is empty
    if st.session_state.df.empty:
        st.warning("No dataset uploaded. Please upload a dataset to view the summary.")
        return  # Exit the function if no data is present

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Shape", "Data Types", "Numerical Data", "Non-Numerical Data", "Missing Values", "Duplicated Rows"])

    with tab1:

        @st.cache_data
        def get_dataframe_info(df):
            return df.shape[0], df.shape[1]

        rows, cols = get_dataframe_info(st.session_state.df)
        st.markdown(f"Rows: {rows:,}  \nColumns: {cols:,}")
                        

    with tab2:
        #
        @st.cache_data
        def get_dataframe_dtypes(df):
            df_dtypes = df.dtypes.reset_index()
            df_dtypes.columns = ['Column', 'Type']
            return df_dtypes

        @st.cache_data
        def group_dtypes(df_dtypes):
            return df_dtypes.groupby('Type')

        # Get and cache the dataframe dtypes
        df_dtypes = get_dataframe_dtypes(st.session_state.df)

        # Group by 'Type' and cache the result
        grouped = group_dtypes(df_dtypes)

        # Display each group separately
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        columns = [col1, col2, col3, col4, col5, col6]

        # Display each group separately
        for i, (dtype, group) in enumerate(grouped):
            with columns[i % 6]:
                st.markdown(f"<h5><span>type:</span> <span style='color: lightgreen;'>{dtype}</span></h5>", unsafe_allow_html=True)
                column_names = group['Column'].tolist()
                st.write("\n".join(f"{j}. {column}" for j, column in enumerate(column_names, 1)))
                st.write("")




    with tab3:

        @st.cache_data
        def get_df_description(df):
            return df.describe()

        st.write(get_df_description(st.session_state.df))

    with tab4:
        # Check if there are columns of type 'object'
        @st.cache_data
        def get_object_columns(df):
            return df.select_dtypes(include='object').columns.tolist()

        @st.cache_data
        def describe_object_columns(df):
            object_columns = get_object_columns(df)
            if object_columns:
                return df[object_columns].describe()
            return None

        object_columns = get_object_columns(st.session_state.df)

        if object_columns:
            description = describe_object_columns(st.session_state.df)
            st.write(description)
        else:
            st.write("No non-numerical data columns to describe.")

    with tab5:

        @st.cache_data
        def get_null_counts(df):
            df_nulls = df.isnull().sum().reset_index()
            df_nulls.columns = ['Column', 'Number of Null Values']
            df_nulls.index = df_nulls.index + 1
            return df_nulls

        df_nulls = get_null_counts(st.session_state.df)
        st.write(df_nulls)

    with tab6:
        #
        @st.cache_data
        def count_duplicates(df):
            return df.duplicated().sum()

        df_duplicated = count_duplicates(st.session_state.df)
        st.write(f"There are {df_duplicated} duplicated rows.")


def fix():
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Convert Data Types", "Handle Missing Values", "Drop Duplicated Rows", "Edit Dataset"])

    with tab1:
        #

        @st.cache_data
        def get_column_names(df):
            return df.columns.tolist()

        @st.cache_data
        def convert_column(df, column_to_convert, new_type, date_format=None):
            df = df.copy()
            try:
                if new_type == "string":
                    df[column_to_convert] = df[column_to_convert].astype(str)
                elif new_type == "datetime":
                    if date_format.lower() == 'mixed':
                        df[column_to_convert] = pd.to_datetime(df[column_to_convert], errors='coerce')
                    else:
                        df[column_to_convert] = pd.to_datetime(df[column_to_convert], format=date_format)
                elif new_type == "integer":
                    df[column_to_convert] = df[column_to_convert].astype(int)
                elif new_type == "float":
                    df[column_to_convert] = df[column_to_convert].astype(float)
                return df, None
            except Exception as e:
                return df, str(e)


        column_names = get_column_names(st.session_state.df)
        column_to_convert = st.selectbox("Select a column to convert", column_names, key = "convert_column")

        data_types = ["string", "datetime", "integer", "float"]
        new_type = st.selectbox("Select the new data type", data_types, key = "convert_type")

        date_format = None
        if new_type == "datetime":
            date_format = st.text_input(
                "Enter the date format (e.g., '%m-%d-%Y', '%Y-%m-%d', or 'mixed' for automatic parsing)",
                value="mixed"
            )

        if st.button("Convert", key="convert_type_bt"):
            st.session_state.df, error = convert_column(st.session_state.df, column_to_convert, new_type, date_format)
            if error:
                st.error(f"Error converting column: {error}")
            else:
                st.success(f"Column '{column_to_convert}' converted to {new_type}")
                backup_df()


    with tab2:
        #

        @st.cache_data
        def get_columns_with_missing(df):
            return df.columns[df.isnull().any()].tolist()

        @st.cache_data
        def apply_missing_value_action(df, column, action, constant_value=None):
            df = df.copy()
            if action == "Drop rows":
                df = df.dropna(subset=[column])
            elif action == "Drop column":
                df = df.drop(columns=[column])
            elif action == "Fill with mean":
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].mean(), inplace=True)
                else:
                    raise ValueError(f"Cannot calculate mean for non-numeric column {column}.")
            elif action == "Fill with median":
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    raise ValueError(f"Cannot calculate median for non-numeric column {column}.")
            elif action == "Fill with mode":
                mode_value = df[column].mode().iloc[0]
                df[column].fillna(mode_value, inplace=True)
            elif action == "Fill with constant":
                df[column].fillna(constant_value, inplace=True)
            return df

        st.markdown('<style>div.stSelectbox > div {width: 20%;}</style>', unsafe_allow_html=True)

        columns_with_missing = get_columns_with_missing(st.session_state.df)

        if not columns_with_missing:
            st.write("No columns with missing values found.")
        else:
            st.write("Select columns and choose how to handle missing values:")
            
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
                    actions[column] = action
                
                if action == "Fill with constant":
                    constant_value = st.text_input(f"Enter constant value for {column}")
                    if constant_value:
                        actions[column] = (action, constant_value)
            
            if actions:
                if st.button("Apply Changes", key="apply_changes_missing_bt"):
                    try:
                        for column, action in actions.items():
                            constant_value = None
                            if isinstance(action, tuple):
                                action, constant_value = action
                            st.session_state.df = apply_missing_value_action(st.session_state.df, column, action, constant_value)
                            st.success(f"Action '{action}' applied to column '{column}'.")
                        
                        st.write("Dataset Updated!")
                        st.dataframe(st.session_state.df.head())
                        backup_df()
                    except Exception as e:
                        st.error(f"Error applying changes: {str(e)}")


    with tab3:
        
        @st.cache_data
        def find_duplicates(df):
            duplicate_rows = df.duplicated()
            num_duplicates = duplicate_rows.sum()
            return num_duplicates, duplicate_rows

        @st.cache_data
        def drop_duplicates(df):
            return df.drop_duplicates()

        # Assuming st.session_state.df is your large DataFrame
        num_duplicates, duplicate_rows = find_duplicates(st.session_state.df)

        if num_duplicates == 0:
            st.write("There are no duplicated rows in the dataset.")
        else:
            st.write(f"There are {num_duplicates} duplicate rows in the dataset.")
            
            if st.button("Drop Duplicates", key="drop_duplicates_bt"):
                st.session_state.df = drop_duplicates(st.session_state.df)
                st.success("All duplicate rows have been dropped.")
                # Assuming backup_df() is defined elsewhere
                backup_df()
    
    with tab4:

        # Initialize session state variables
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0

        # Column selection
        search_column = st.selectbox("Select column to search", st.session_state.df.columns, key = "df_search_column")

        # Add a checkbox for wildcard search
        use_wildcard = st.checkbox("Use wildcard search")


        if use_wildcard:

            st.info("""
                Wildcard Search Examples:
                - Use * to match any number of characters
                - Use ? to match a single character
                
                Examples:
                - "50*" matches: 50, 500, 5000
                - "?50" matches: 150, 250, 350
                - "5?0" matches: 500, 510, 520
                """)


        # Search input
        search_term = st.text_input("Enter search term", key = "search_df")

        # Search button
        if st.button("Search"):
            if search_term:
                if use_wildcard:
                    # Convert wildcard pattern to regex
                    search_pattern = search_term.strip().replace("*", ".*").replace("?", ".")
                    regex = re.compile(f"^{search_pattern}$", re.IGNORECASE)
                    filtered_df = st.session_state.df[
                        st.session_state.df[search_column].apply(
                            lambda x: bool(regex.match(str(x)))  # Ensure all values are treated as strings
                        )
                    ]
                else:
                    # Exact match search with normalization
                    def normalize(value):
                        try:
                            # Try converting to float and compare with the search term as float
                            return float(value) == float(search_term)
                        except ValueError:
                            # Fallback to string comparison if conversion fails
                            return str(value) == search_term
                    
                    filtered_df = st.session_state.df[
                        st.session_state.df[search_column].apply(normalize)
                    ]
                
                st.session_state.search_results = filtered_df.index.tolist()
                st.session_state.current_index = 0
                st.rerun()
            else:
                st.warning("Please enter a search term")



        # Display and edit results one by one
        if st.session_state.search_results:
            st.write(f"Result {st.session_state.current_index + 1} of {len(st.session_state.search_results)}")
            
            current_row_index = st.session_state.search_results[st.session_state.current_index]
            current_row = st.session_state.df.loc[[current_row_index]]
            
            edited_row = st.data_editor(current_row, key=f"editor_{st.session_state.current_index}")
            
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                if st.button("Previous", use_container_width=True) and st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
                    st.rerun()

            with col2:
                if st.button("Save Changes", use_container_width=True):
                    # Retrieve the edited row
                    edited_value = edited_row.iloc[0, :]
                    
                    # Enforce float conversion for numeric columns
                    if pd.api.types.is_numeric_dtype(st.session_state.df[search_column]):
                        try:
                            # Convert to float and add `.0` if it's an integer
                            edited_value = edited_value.apply(
                                lambda x: float(x) if isinstance(x, (int, str)) and str(x).isdigit() else x
                            )
                        except ValueError:
                            st.warning("Invalid input: Unable to convert to float.")
                            st.stop()
                    
                    # Update the dataframe with the modified value
                    st.session_state.df.loc[current_row_index] = edited_value
                    st.success("Changes saved to the original dataset")
                    st.rerun()



            with col3:
                if st.button("Next", use_container_width=True) and st.session_state.current_index < len(st.session_state.search_results) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
            

        else:
            st.info("No search results to display. Please perform a search.")


def new_columns():
    
    tab1, tab2, tab3 = st.tabs(["Add New Columns", "Delete Columns", "Rename Columns"])

    with tab1:
        # Sidebar for managing functions
        with st.sidebar:
            st.header("Manage Custom Functions")
            
            # Add/Edit function
            st.subheader("Add/Edit Function")
            st.text_input("Function Name", key="function_name")
            function_code = st_ace(
                placeholder="Write your Python function here",
                language="python",
                theme="monokai",
                keybinding="vscode",
                font_size=14,
                tab_size=4,
                show_gutter=True,
                show_print_margin=False,
                wrap=False,
                auto_update=True,
                key="function_code"
            )

            st.button("Save Function", on_click=add_or_update_function)
            
            # Remove function
            st.subheader("Remove Function")
            func_to_remove = st.selectbox("Select function to remove", 
                                          options=list(st.session_state.custom_functions.keys()),
                                          key="function_to_remove")
            st.button("Remove Function", on_click=remove_function)


        # Add a new column

        new_col_name = st.text_input("Enter new column name:", key = "new_column")
        new_col_value = st.text_input("Enter value or formula (use `df['column_name']` for existing columns):", key = "column_formula")
        st.write("Add your custom function from the sidebar and use it like this: `df['column_name'].apply(custom_function)`.")

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
        #
        st.markdown('<style>div.stMultiSelect > div {width: 30%;}</style>', unsafe_allow_html=True)

        # Allow selection of all columns, not just new ones
        columns_to_delete = st.multiselect("Select columns to delete:", st.session_state.df.columns, key="delete_columns")

        if st.button("Delete", key="delete_columns_bt"):
            if columns_to_delete:
                st.session_state.df.drop(columns=columns_to_delete, inplace=True)
                st.success(f"Deleted columns: {', '.join(columns_to_delete)}")
                st.write(st.session_state.df.head())
                # Reset the multiselect after deletion
                st.session_state.columns_to_delete = []
                backup_df()  # Assuming this function exists to backup the dataframe
            else:
                st.warning("No columns selected for deletion.")

     


    with tab3:

        # Rename columns
        # Get list of current column names
        current_columns = st.session_state.df.columns.tolist()
        
        # Select column to rename
        column_to_rename = st.selectbox("Select column to rename:", current_columns, key = "rename_columns")
        
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
    
    col1, col2 = st.columns(2)
    with col1:

        with st.container(border = True):
            st.subheader("Local")

            @st.cache_data
            def process_json_file(file):
                try:
                    content = file.getvalue()
                    settings_json = content.decode("utf-8")
                    return settings_json
                except UnicodeDecodeError:
                    st.error("Invalid file encoding. Please upload a UTF-8 encoded JSON file.")
                    return None

            # File uploader for importing settings
            json_file = st.file_uploader("Choose your local dashboard file", type="json")

            if json_file is not None:
                st.session_state.json_file = json_file
                settings_json = process_json_file(json_file)
                
                if settings_json:
                    try:
                        import_settings(settings_json)  # Passing the JSON string directly
                        
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format. Please upload a valid JSON file.")
                    except Exception as e:
                        st.error(f"Error importing settings: {str(e)}")


            # Ensure the dataframe is not empty
            df_name = st.session_state.df_name
            

            @st.cache_data
            def load_and_prepare_data(df):
                csv_data = df.to_csv(index=False).encode('utf-8')
                return csv_data

            @st.cache_data
            def prepare_dashboard_json():
                return export_settings()

            # Prepare data once and store in session state
            if 'prepared_csv_data' not in st.session_state and 'df' in st.session_state and not st.session_state.df.empty:
                with st.spinner('Preparing data for download...'):
                    st.session_state.prepared_csv_data = load_and_prepare_data(st.session_state.df)
                    st.session_state.prepared_json_data = prepare_dashboard_json()

            # Display download buttons if data is prepared
            if 'prepared_csv_data' in st.session_state:
                dataset_file_name = df_name.rsplit(".", 1)[0] + "_updated.csv"
                st.download_button(
                    label="Download Dataset",
                    data=st.session_state.prepared_csv_data,
                    file_name=dataset_file_name,
                    mime="text/csv"
                )

                dashboard_file_name = df_name.rsplit(".", 1)[0] + "_dashboard.json"
                st.download_button(
                    label="Download Dashboard",
                    data=st.session_state.prepared_json_data,
                    file_name=dashboard_file_name,
                    mime="application/json"
                )

    with col2:

        with st.container(border = True):

            st.subheader("Cloud")
           
            # Initialize the S3 client
            @st.cache_resource
            def get_s3_client():
                return boto3.client(
                    's3',
                    aws_access_key_id='AKIAQUFLQN6S3NYTLU7Q',
                    aws_secret_access_key='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',
                    region_name='ap-southeast-2'
                )

            s3 = get_s3_client()

            bucket_name = 'dataphil-bucket'

            settings_json = export_settings()



            def upload_file_to_s3(data, bucket_name):
                try:
                    unique_filename = f"dashboard_{uuid.uuid4().hex}.json"
                    file_obj = io.BytesIO(data.encode('utf-8'))
                    file_size = file_obj.getbuffer().nbytes
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Implement multipart upload
                    mpu = s3.create_multipart_upload(Bucket=bucket_name, Key=unique_filename)
                    
                    parts = []
                    uploaded_bytes = 0
                    part_size = 5 * 1024 * 1024  # 5MB chunks
                    
                    for i, chunk in enumerate(iter(lambda: file_obj.read(part_size), b'')):
                        part = s3.upload_part(Body=chunk, Bucket=bucket_name, Key=unique_filename, PartNumber=i+1, UploadId=mpu['UploadId'])
                        parts.append({"PartNumber": i+1, "ETag": part['ETag']})
                        uploaded_bytes += len(chunk)
                        progress_bar.progress(min(uploaded_bytes / file_size, 1.0))
                    
                    s3.complete_multipart_upload(Bucket=bucket_name, Key=unique_filename, UploadId=mpu['UploadId'], MultipartUpload={"Parts": parts})
                    
                    file_url = f"https://{bucket_name}.s3.ap-southeast-2.amazonaws.com/{unique_filename}"
                    return unique_filename
                except Exception as e:
                    return f"Error: {e}"

            if "df_hash" not in st.session_state:
                st.session_state.df_hash = calculate_df_hash(st.session_state.df)


            if "dashboard_hash" not in st.session_state:
                st.session_state.dashboard_hash = calculate_dashboard_hash(
                    st.session_state.layout, st.session_state.charts
                )


            if st.button("Upload dashboard"):
                current_hash = calculate_df_hash(st.session_state.df)
                current_dashboard_hash = calculate_dashboard_hash(
                    st.session_state.layout, st.session_state.charts
                )

                if current_hash != st.session_state.df_hash or current_dashboard_hash != st.session_state.dashboard_hash:
                    with st.spinner('Uploading...'):
                        result = upload_file_to_s3(settings_json, bucket_name)
                    result = result.replace('.json', '')

                    # Update hashes after successful upload
                    if "Error" not in result:
                        st.session_state.df_hash = current_hash
                        st.session_state.dashboard_hash = current_dashboard_hash
                        st.success("File uploaded successfully!")

                        st.write("Your dashboard code:", result)
                    else:
                        st.error(f"An error occurred: {result}")
                else:
                    st.info("No changes detected in the dataset. File upload skipped.")



            def download_file_from_s3(bucket_name, file_name):
                try:
                    # Get file metadata
                    response = s3.head_object(Bucket=bucket_name, Key=file_name)
                    file_size = response['ContentLength']
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Stream the file
                    streamed_body = s3.get_object(Bucket=bucket_name, Key=file_name)['Body']
                    
                    chunk_size = 1024 * 1024  # 1MB chunks
                    downloaded_bytes = 0
                    file_content = io.BytesIO()
                    
                    for chunk in streamed_body.iter_chunks(chunk_size=chunk_size):
                        file_content.write(chunk)
                        downloaded_bytes += len(chunk)
                        progress_bar.progress(min(downloaded_bytes / file_size, 1.0))
                    
                    file_content.seek(0)
                    return file_content
                except Exception as e:
                    return f"Error: {e}"



            st.markdown(
                """
                <style>
                .stTextInput {
                    width: 100%; 
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            filename = st.text_input("Enter dashboard code:")
            if filename and st.button("Download from the Cloud"):
                json_file = download_file_from_s3("dataphil-bucket", f"{filename}.json")
                
                if isinstance(json_file, io.BytesIO):
                    # Decode and process the file content
                    settings_json = json_file.getvalue().decode("utf-8")
                    st.session_state.json_file = json_file
                    import_settings(settings_json)
                    st.write("Dashboard loaded successfully!")
                else:
                    st.write("An error occurred:", json_file)


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
            df = st.session_state.df
            st.session_state.selected_df = df


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



def report():
     tab1, tab2, tab3 = st.tabs(["Tables", "Filters", "Visualization"])
 
     with tab1:
        #Tables
        st.markdown('<style>div.stSelectbox > div {width: 80%;}</style>', unsafe_allow_html=True)
        st.markdown('<style>div.stMultiSelect > div {width: 60%;}</style>', unsafe_allow_html=True)


        
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
     
     with tab2:
        #Filters
        st.markdown('<style>div.stSelectbox > div {width: 80%;}</style>', unsafe_allow_html=True)

        
        # Step 1: Select dataframe
        dataframe_options = ["Original Dataframe"] + list(st.session_state.get('tb', {}).keys())
        selected_df_name = st.selectbox("Select a dataframe:", dataframe_options, key = "df_filter")
        
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
        group_columns = st.multiselect("Select columns to group by:", st.session_state.df.columns, key = "aggregate_columns")

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
            agg_column = st.selectbox("Select column:", st.session_state.df.columns, key = "agg_column")
        with col2:
            agg_function = st.selectbox("Select function:", agg_functions, key = "agg_function")
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
            del st.session_state.agg_result
            st.session_state.agg_result = result

            st.session_state.agg_code = f'df.groupby({group_columns}).agg({agg_dict}).reset_index()'
            
            # Display the result
            st.write("Aggregated Report:")
            st.dataframe(result)
            
            
            return st.session_state.agg_code



def dashboard_tab():



    # Check for saved results
    st.session_state.tb = st.session_state.get('tb', {})
    tb = pd.DataFrame(st.session_state.tb.items(), columns=['Key', 'Value'])

    st.session_state.custom_title = st.text_input("Enter dashboard title:", "Dashboard", key = "tui_title")

    with st.container(border = True):
        # Let the user define the dashboard layout
        rows = st.number_input("Number of rows", min_value=1, max_value = 10, value=2, key = "dash_row")
        cols = st.number_input("Number of columns", min_value=1, max_value = 10, value=2, key= "dash_col")

    # Create a list of cell positions
    cell_positions = [f"{i+1}-{j+1}" for i in range(rows) for j in range(cols)]

    # Store the layout in session state
    if "layout" not in st.session_state or st.session_state.layout != {"rows": rows, "cols": cols, "cells": cell_positions}:
        st.session_state.layout = {"rows": rows, "cols": cols, "cells": cell_positions}
        st.session_state.charts = {}  # Reset charts on layout change

    df = st.session_state.df
    st.dataframe(df.head(5))

   
    
    tab1, tab2 = st.tabs(['TUI', 'GUI'])

    with tab1:#TUI

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
        selected_chart = st.selectbox("Select a chart type", chart_list, key = "tui_chart_type")

        col1, col2, col3 = st.columns([3,1,3], gap="small")
        with col1:
            # Let the user input their own code
            user_code = st_ace(

                value=f'agg_data = {st.session_state.agg_code}\n{st.session_state.chart_code}',
                language="python",
                theme="monokai",
                keybinding="vscode",
                min_lines=10,
                max_lines=20,
                font_size=14,
                tab_size=4,
                show_gutter=True,
                show_print_margin=False,
                wrap=False,
                auto_update=True,
                key="user_code_editor"
            )
                        

            st.markdown("Check out [seaborn](https://seaborn.pydata.org/examples/index.html), [matplotlib](https://matplotlib.org/stable/gallery/index.html), [plotly](https://plotly.com/python/), and [altair](https://altair-viz.github.io/gallery/index.html).")

            

        with col2:
            column_types = pd.DataFrame({'Data Types': df.dtypes.astype(str)})
            st.dataframe(column_types, width=500)
        with col3:
            # Display sample code for the selected chart type
            if selected_chart:
                sample_code = generate_chart_code(selected_chart)
                st.code(sample_code, language='python')



    with tab2: #GUI


        
        #

        aggregate_choice = st.radio("Do you want to aggregate the dataset?", ("Yes", "No"))
        if aggregate_choice == "Yes":
            
            st.session_state.agg_code = aggregate()
            st.session_state.agg_event = True
            df = st.session_state.agg_result

        else:
            st.session_state.agg_event = False
            st.write("No aggregation performed.")


        with st.container(key = "gui_elements"):

            # Chart Type Selection
            chart_options = [
                    "None", "Area Chart", "Bar Chart", "Line Chart", "Scatter Chart",
                    "Map", "Pie Chart", "Histogram", "Box Plot", "Heatmap",
                    "Violin Chart", "Bubble Chart", "Sunburst Chart", "Treemap",
                    "Streamgraph", "Candlestick Chart", "Radar Chart", "WordCloud",
                    "Timeline Chart", "Density Chart", "Gauge Chart", "KPI Card", "Text","Image","Video"
                ]

            st.session_state["chart_type"] = st.selectbox(
                "Select the type of chart you want to create",
                chart_options,
                index=0 if st.session_state.get("chart_type") is None else chart_options.index(st.session_state["chart_type"]),
                placeholder="Choose a chart type...", key = "gui_chart"
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
                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f'''st.area_chart(
                    data = {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'
                
                st.session_state.chart_code = textwrap.dedent(f"""st.line_chart(
                    data = {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'


                st.session_state.chart_code = textwrap.dedent(f'''st.scatter_chart(
                    data = {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'



                st.session_state.chart_code = textwrap.dedent(f"""st.map(
                    data = {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'


                st.session_state.chart_code = textwrap.dedent(f"""
                    fig = px.pie(
                    {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
                    names = {repr(names)},
                    values = {repr(values)},
                    )
                    st.plotly_chart(fig)
                """)

            
            if chart_type == "Histogram":
                x = st.selectbox("X", df.columns)
                color = st.selectbox("Color", [None]+list(df.columns))
                use_container_width = st.checkbox("use container width", value=True)

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                    fig = px.histogram(
                    {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                    fig = px.box(
                        {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation_matrix = {data[1:-1] if data.startswith("'") and data.endswith("'") else data}[{repr(selected_columns)}].corr(method={repr(corr_method)})
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                    fig = px.violin(
                        {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                    fig = px.scatter(
                        {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                if {repr(path_columns)} and {repr(value_column)}:
                    fig = px.sunburst(
                        {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                    if {repr(path_columns)} and {repr(value_column)}:
                        fig = px.treemap(
                            {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                    streamgraph = alt.Chart({data[1:-1] if data.startswith("'") and data.endswith("'") else data}).mark_area().encode(
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

                    if st.session_state.agg_event:
                        data = 'agg_data'
                    else:
                        data = 'df'

                    st.session_state.chart_code = textwrap.dedent(f"""
                        df_filtered = {data[1:-1] if data.startswith("'") and data.endswith("'") else data}[({data[1:-1] if data.startswith("'") and data.endswith("'") else data}[{repr(date_column)}] >= str({repr(date_range[0])})) & ({data[1:-1] if data.startswith("'") and data.endswith("'") else data}[{repr(date_column)}] <= str({repr(date_range[1])}))]

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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                    fig = px.line_polar(
                        {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                text = ' '.join({data[1:-1] if data.startswith("'") and data.endswith("'") else data}[{repr(text_column)}].dropna().astype(str))
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                    fig = px.timeline(
                        {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                    fig = px.density_contour(
                        {data[1:-1] if data.startswith("'") and data.endswith("'") else data},
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

                    if st.session_state.agg_event:
                        data = 'agg_data'
                    else:
                        data = 'df'

                    st.session_state.chart_code = textwrap.dedent(f"""
                        gauge_value = {data[1:-1] if data.startswith("'") and data.endswith("'") else data}[{repr(value_column)}].iloc[-1]

                        reference_value = None
                        if {repr(reference_column)} != "None":
                            reference_value = {data[1:-1] if data.startswith("'") and data.endswith("'") else data}[{repr(reference_column)}].iloc[-1]

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

                if st.session_state.agg_event:
                    data = 'agg_data'
                else:
                    data = 'df'

                st.session_state.chart_code = textwrap.dedent(f"""
                    # Get the latest values from the selected columns
                    label = {data[1:-1] if data.startswith("'") and data.endswith("'") else data}[{repr(label_column)}].iloc[-1]
                    value = {data[1:-1] if data.startswith("'") and data.endswith("'") else data}[{repr(value_column)}].iloc[-1]
                    delta = {data[1:-1] if data.startswith("'") and data.endswith("'") else data}[{repr(delta_column)}].iloc[-1] if {repr(delta_column)} != "None" else None

                    st.metric(
                        label=label,
                        value=value,
                        delta=delta,
                        delta_color={repr(delta_color)},
                        label_visibility={repr(label_visibility)},
                        border={repr(border)}
                    )
                    """)



            if chart_type == "Text":

                # Text input
                user_text = st.text_input("Enter your text:")

                # Font size selection
                font_size = st.slider("Select font size:", 10, 30, 16)

                # Text style selection
                bold = st.checkbox("Bold")
                italic = st.checkbox("Italic")

                # Hyperlink option
                add_link = st.checkbox("Add a hyperlink")

                if add_link:
                    link_text = st.text_input("Enter the text to be linked:")
                    link_url = st.text_input("Enter the URL:")
                    if not link_url.startswith(('http://', 'https://')):
                        link_url = 'https://' + link_url

                # Display the styled text
                if user_text:
                    style = f"font-size: {font_size}px;"
                    if bold:
                        style += " font-weight: bold;"
                    if italic:
                        style += " font-style: italic;"
                    
                    if add_link and link_text and link_url:
                        user_text = user_text.replace(link_text, f"<a href='{link_url}'>{link_text}</a>")
                    
                    st.session_state.chart_code = textwrap.dedent(f"""
                        st.markdown(f"<p style='{style}'>{user_text}</p>", unsafe_allow_html=True)
                        """)

                    # Display the styled text
                    st.markdown(f"<p style='{style}'>{user_text}</p>", unsafe_allow_html=True)


            if chart_type == "Image":

                # Image link input
                image_link = st.text_input("Enter image URL:")

                if image_link:
                    # Image width slider
                    image_width = st.slider("Adjust image width:", 100, 800, 400)
                    
                    # Display the image with adjusted width
                    st.session_state.chart_code = textwrap.dedent(f"""
                        st.image("{image_link}", width={image_width})
                        """)


            if chart_type == "Video":

                # Video link input
                video_link = st.text_input("Enter YouTube video URL:")
                if video_link:
                    st.session_state.chart_code = textwrap.dedent(f"""
                        st.video("{video_link}")
                        """)


    # Ask for chart title and axis labels
    chart_title = st.text_input("Chart title", key = "chart_title")

    # Let the user select the cell position
    selected_cell = st.selectbox("Select cell position", st.session_state.layout["cells"], key="select_cell")

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
                    st.markdown('<style>div.stSelectbox > div {width: 100%;}</style>', unsafe_allow_html=True)

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


    # Add a toggle button to show or hide the adjustment interface
    if "show_adjustments" not in st.session_state:
        st.session_state.show_adjustments = False

    if st.button("Adjustments"):
        st.session_state.show_adjustments = not st.session_state.show_adjustments 

    if st.session_state.show_adjustments:
      adjust_dashboard_layout()



    layout_config = st.session_state.layout
    num_rows = layout_config["rows"]
    num_cols = layout_config["cols"]

    # Ensure column_widths is properly initialized
    column_widths = st.session_state.get("column_widths", [[50] * num_cols for _ in range(num_rows)])

    # Render the dashboard
    for row_index in range(num_rows):
        if row_index >= len(column_widths):
            # Handle missing rows in column_widths gracefully
            st.warning(f"Row {row_index + 1} is missing in column_widths. Using default widths.")
            row_widths = [50] * num_cols
        else:
            row_widths = column_widths[row_index]

        # Normalize widths
        if sum(row_widths) == 0:
            st.error(f"Invalid widths in row {row_index + 1}. All values are zero.")
            normalized_widths = [12 // num_cols] * num_cols  # Equal width as fallback
        else:
            normalized_widths = [int((w / sum(row_widths)) * 12) for w in row_widths]

        # Create columns for the current row
        row_columns = st.columns(normalized_widths)

    

        for col_index, col in enumerate(row_columns):

                cell = f"{row_index+1}-{col_index+1}"
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




# Show dashboard toggle
if show_only_dashboard:
    dashboard()
else:



    # header

    with st.container(key = "logo"):
        st.image("img/dataphil_logo.png", width=150)

    st.markdown("<h6 style='text-align: center;'>DataPhil: Your Data, Your Way</h6>", 
            unsafe_allow_html=True)
    # sections
    with st.container(key = "sections"):
        section_selection = st.pills("", ["Upload Dataset", "Summary", "Fix Dataset", "New Columns", "Import/Export", "Report", "Dashboard"])
    # Display content based on sidebar selection
    if section_selection == "Upload Dataset":
        upload_dataset()

    elif section_selection == "Summary":
        summary()

    elif section_selection == "Fix Dataset":
        fix()

    elif section_selection == "New Columns":
        new_columns()

    elif section_selection == "Import/Export":
        export()

    elif section_selection == "Report":
        report()
    elif section_selection == "Dashboard":
        dashboard_tab()