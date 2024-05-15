from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
import streamlit as st

from few_shots import few_shots

import os
from dotenv import load_dotenv
load_dotenv()

# comment the marked snippet if using env file for secrets
#----------------------------------- 
os.environ["GOOGLE_API_KEY"] == st.secrets["GOOGLE_API_KEY"]
os.environ["DB_PASSWORD"] == st.secrets["DB_PASSWORD"]
#-------------------------------------------

def get_few_shot_db_chain():
    db_user = "sql12706690"
    db_host = "sql12.freesqldatabase.com"
    db_name = "sql12706690"
    db_port = 3306

    db = SQLDatabase.from_uri("mysql+pymysql://" + db_user + ":" + os.environ["DB_PASSWORD"] + "@" + db_host + ":" + str(db_port) + "/" + db_name, sample_rows_in_table_info=3)

    llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    
    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=few_shots,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"]
    )
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain

