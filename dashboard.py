# import modules

import pandas as pd
import streamlit as st
import numpy as np
import os


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

if 'saved_results' not in st.session_state:
    st.session_state.saved_results = {}

if 'filters' not in st.session_state:
    st.session_state.filters = []


# backup of session
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

    tab1, tab2 = st.tabs(["Add New Columns", "Delete Columns"])

    with tab1:
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
            for col_name, col_value in st.session_state.new_columns:
                try:
                    st.session_state.df[col_name] = eval(col_value, {'df': st.session_state.df, 'pd': pd, 'np': np})
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

def export():
    restore_backup()

    # Ensure the dataframe is not empty
    if 'df' in st.session_state and not st.session_state.df.empty:
        # Check if uploaded_file exists in session state
        if 'uploaded_file' in st.session_state:
            uploaded_file = st.session_state.uploaded_file
            # Generate the updated file name
            file_name = "updated_" + uploaded_file  # Adding _updated to the original filename
            file_name = file_name.replace(".csv", "_updated.csv")  # Ensure it has the .csv extension
            
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
            st.error("No file uploaded. Please upload a dataset first.")

def upload_dataset():

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

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

# Function to generate chart code based on user input
def generate_chart_code(chart_type, df_name='df'):
    if chart_type in ["area_chart", "bar_chart", "line_chart"]:
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
    elif chart_type == "scatter_chart":
        return f"""
        import altair as alt

        scatter_plot = alt.Chart({df_name}).mark_point().encode(
            x='x_column',
            y='y_column',
            color='color_column'
        )
        st.altair_chart(scatter_plot, use_container_width=True)
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
                    if 'saved_results' not in st.session_state:
                        st.session_state.saved_results = {}
                    st.session_state.saved_results[save_name] = result
                    st.success(f"Result saved as '{save_name}'")
                else:
                    st.warning("Please enter a name to save the result.")

            # Display saved results
            if 'saved_results' in st.session_state and st.session_state.saved_results:
                st.write("Saved Results:")
                for name in st.session_state.saved_results.keys():
                    st.write(f"- {name}")
        else:
            st.write("Please select grouping columns and add at least one aggregation.")
     
     with tab2: #Filters
        restore_backup()
        # Step 1: Select dataframe
        dataframe_options = ["Original Dataframe"] + list(st.session_state.get('saved_results', {}).keys())
        selected_df_name = st.selectbox("Select a dataframe:", dataframe_options)
        
        if selected_df_name == "Original Dataframe":
            df = st.session_state.df
        else:
            df = st.session_state.saved_results[selected_df_name]
        
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
                    if 'saved_results' not in st.session_state:
                        st.session_state.saved_results = {}
                    st.session_state.saved_results[save_name] = st.session_state.filtered_df
                    st.success(f"Filtered result saved as '{save_name}'")
                else:
                    st.warning("Please enter a name to save the filtered result.")

        # Display saved results
        if 'saved_results' in st.session_state and st.session_state.saved_results:
            st.write("Saved Results:")
            for name in st.session_state.saved_results.keys():
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

def dashboard_tab():

        restore_backup()

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

        # Allow the user to select available dataframes
        available_dfs = ['df']  # Start with 'df' as an option

        if 'saved_results' in st.session_state and isinstance(st.session_state.saved_results, dict):
            available_dfs.extend(list(st.session_state.saved_results.keys()))

        if available_dfs:
            selected_df = st.selectbox("Select a dataframe", available_dfs)

            # Use the selected dataframe
            if selected_df == 'df':
                df = st.session_state.df
            elif selected_df in st.session_state.saved_results:
                df = st.session_state.saved_results[selected_df]
            else:
                st.error(f"The selected dataframe '{selected_df}' is not available in session state.")
                st.stop()

            st.dataframe(df.head())
            st.session_state.selected_df = df


            # List of available chart types
            chart_list = [
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

            # Display sample code for the selected chart type
            if selected_chart:
                sample_code = generate_chart_code(selected_chart)
                st.code(sample_code, language='python')

            # Let the user input their own code
            user_code = st.text_area("Enter your custom code for the chart:", height=200)

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
        st.button("Add Filter", on_click=add_filter)

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
        if st.button("Update Dashboard"):
            df = apply_filters(df)




    if "rows" not in st.session_state.layout or "cols" not in st.session_state.layout:
        st.error("Dashboard layout is not configured. Please set it up in the Report section first.")
        return

    # Allow user to enter a custom title only if the toggle is off
    if not show_only_dashboard:
        custom_title = st.text_input("Enter dashboard title:", "Dashboard")
        st.session_state.custom_dashboard_title = custom_title

    else:
        # Retrieve the stored title from session state, or default to "Dashboard"
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
                        exec(chart_data["code"], {"df": df, "st": st})
                    except Exception as e:
                        st.error(f"Error executing custom code: {str(e)}")
                        st.error(f"Chart data: {chart_data}")


# Show the dashboard if toggle is active
if show_only_dashboard:
    dashboard()
else:
    # header
    st.markdown("<h2 style='text-align: center;'>Welcome to DataPhil!👋</h2>",
            unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'><i>Your Data, Your Way—Fast & Easy!</i></h6>", 
            unsafe_allow_html=True)

    # sections
    section_selection = st.pills("", ["Upload Dataset", "Summary", "Fix Dataset", "New Columns", "Export", "Report", "Dashboard"])
    # Display content based on sidebar selection
    if section_selection == "Upload Dataset":
        upload_dataset()

    elif section_selection == "Summary":
        summary()

    elif section_selection == "Fix Dataset":
        fix()

    elif section_selection == "New Columns":
        new_columns()

    elif section_selection == "Export":
        export()

    elif section_selection == "Report":
        report()
    elif section_selection == "Dashboard":
        dashboard_tab()
