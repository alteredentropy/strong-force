import os
import re
import json
from datetime import datetime
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, ProgrammingError
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Sequence, Optional, Dict, Literal
import yaml

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination, HandoffTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat, Swarm, MagenticOneGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.messages import AgentEvent, ChatMessage, HandoffMessage
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import CancellationToken
from autogen_core import Image
from autogen_core.tools import FunctionTool
# Load environment variables from the .env file
load_dotenv()


# ------------------------------------------------------------------------------
# OpenAI Integration
# ------------------------------------------------------------------------------


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is required for Autogen integration.")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")


# ------------------------------------------------------------------------------
# Flask Application Setup
# ------------------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------------------
# Database Configuration & Schema Reflection
# ------------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/mydb")
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Global in-memory schema info: {table_name: [column1, column2, ...]}
SCHEMA_INFO = {}

def load_schema():
    """
    Reflect the database schema and load table and column information into SCHEMA_INFO.
    """
    global SCHEMA_INFO
    try:
        metadata.reflect(bind=engine)
        for table in metadata.tables.values():
            SCHEMA_INFO[table.name] = list(table.columns.keys())
        app.logger.info("Database schema loaded: %s", json.dumps(SCHEMA_INFO))
    except SQLAlchemyError as e:
        app.logger.error("Error loading schema: %s", str(e))
        SCHEMA_INFO = {}

# ------------------------------------------------------------------------------
# Business Logic Configuration
# ------------------------------------------------------------------------------

# Path to your YAML file
rules_path = os.path.join(os.path.dirname(__file__), "config", "rules.yaml")

with open(rules_path, "r") as f:
    BUSINESS_RULES = yaml.safe_load(f)

# ------------------------------------------------------------------------------
# Model Tooling
# ------------------------------------------------------------------------------

# TODO: Should return an enum of either SELECT, DELETE, OR UPDATE
def extract_operation(sql: str)-> str:
    """
    Extract the SQL operation (first word) from the given SQL statement.
    """
    if not sql:
        return ""
    return sql.strip().split()[0].upper()


def extract_table_name(sql: str, operation: str) -> dict:
    """
    Extract the target database, schema, and table name from the SQL statement using regex.
    Supports three forms:
      1. table
      2. schema.table
      3. database.schema.table
    for SELECT, INSERT, UPDATE, and DELETE statements.
    
    Returns:
        A dictionary with keys 'database', 'schema', and 'table'. For example:
          { "database": "mydb", "schema": "public", "table": "customers" }
        If a part is not provided, its value will be None.
        If no match is found, returns None.
    """
    pattern = None
    op = operation.upper()
    if op == "SELECT":
        pattern = (r"FROM\s+"
                   r"(?:(?P<database>[a-zA-Z_][a-zA-Z0-9_]*)\.)?"  # optional database
                   r"(?:(?P<schema>[a-zA-Z_][a-zA-Z0-9_]*)\.)?"    # optional schema
                   r"(?P<table>[a-zA-Z_][a-zA-Z0-9_]*)")            # required table
    elif op == "INSERT":
        pattern = (r"INSERT\s+INTO\s+"
                   r"(?:(?P<database>[a-zA-Z_][a-zA-Z0-9_]*)\.)?"
                   r"(?:(?P<schema>[a-zA-Z_][a-zA-Z0-9_]*)\.)?"
                   r"(?P<table>[a-zA-Z_][a-zA-Z0-9_]*)")
    elif op == "UPDATE":
        pattern = (r"UPDATE\s+"
                   r"(?:(?P<database>[a-zA-Z_][a-zA-Z0-9_]*)\.)?"
                   r"(?:(?P<schema>[a-zA-Z_][a-zA-Z0-9_]*)\.)?"
                   r"(?P<table>[a-zA-Z_][a-zA-Z0-9_]*)")
    elif op == "DELETE":
        pattern = (r"DELETE\s+FROM\s+"
                   r"(?:(?P<database>[a-zA-Z_][a-zA-Z0-9_]*)\.)?"
                   r"(?:(?P<schema>[a-zA-Z_][a-zA-Z0-9_]*)\.)?"
                   r"(?P<table>[a-zA-Z_][a-zA-Z0-9_]*)")
    
    if pattern:
        match = re.search(pattern, sql, re.IGNORECASE)
        if match:
            groups = match.groupdict()
            return {
                "database": groups.get("database", None).lower() if groups.get("database") else None,
                "schema": groups.get("schema", None).lower() if groups.get("schema") else None,
                "table": groups.get("table", None).lower() if groups.get("table") else None,
            }
    return None

extract_operation_tool = FunctionTool(extract_operation, name="extract_operation", description="Extracts the operation type from a SQL query.", strict=True)
extract_table_name_tool = FunctionTool(extract_table_name, name="extract_table_name", description="Extracts the table name from a SQL query.", strict=True)

# ------------------------------------------------------------------------------
# Agents
# ------------------------------------------------------------------------------

async def create_agent_sql_parcer():

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        response_format=AgentResponseTableName,
        api_key=OPENAI_API_KEY,
        organization=OPENAI_ORG_ID
        )

    agent = AssistantAgent(
        name="SQL Part Extractor",
        model_client=model_client,
        tools=[extract_operation_tool,extract_table_name_tool],
        description="Extracts out different parts of a valid PostgreSQL SQL query",
        system_message="""You are an AI assistant that Extracts out different parts of a valid PostgreSQL SQL query.
        You can extract the statement type and the full table name with the use of your tools.
        """,
        )

    return agent

async def create_agent_nl_parcer():

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        organization=OPENAI_ORG_ID
        )

    agent = AssistantAgent(
        name="Natural Language to SQL Parcer",
        model_client=model_client,
        #tools=[extract_operation_tool,extract_table_name_tool],
        description="Converts natural language requests into valid PostgreSQL SQL queries",
        system_message="""You are an AI assistant that converts natural language requests into valid PostgreSQL SQL queries.
        """,
        )

    return agent


async def create_agent_verify_sql():

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        organization=OPENAI_ORG_ID
        )

    agent = AssistantAgent(
        name="Valid SQL Checker",
        model_client=model_client,
        #tools=[extract_operation_tool,extract_table_name_tool],
        description="Checks the incomming message and verifies if its valid SQL or not",
        system_message="""You are an AI assistant that checks the incomming message and verifies if its valid SQL or not.
        """,
        )

    return agent


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


async def parse_nl_to_sql(nl_query):
    """
    Use Autogen to convert a natural language query into a valid PostgreSQL SQL statement.
    
    Parameters:
        nl_query (str): The natural language request.
                            
    Returns:
        sql (str): The generated SQL statement.
        
    Raises:
        ValueError: If the AI fails to parse the request or the generated operation is not allowed.
    """


    agent = await create_agent_nl_parcer()
    sql=None
    try:
        response = await agent.run(task=f"""
                Convert this natural language statement into a valid SQL statement .Do not include markup and return only the sql statement.
                Language statement: {nl_query}
                """)
        sql = response.strip()
    except Exception as e:
        raise ValueError("Autogen parsing failed: " + str(e))
    return sql


async def parse_sql(sql):

    agent = await create_agent_sql_parcer()
    try:
        response = await agent.run(task=f"""
                Extract out the Statement type (select, update, or delete) and the source table name.
                SQL statement: {sql}
                """)
    except Exception as e:
        raise ValueError("Autogen parsing failed: " + str(e))
    return response







def apply_business_rules(sql, query_type):
    """
    Apply business logic rules to the generated SQL.
    
    Parameters:
        sql (str): The SQL statement.
        query_type (str): "read" or "write".
        
    Returns:
        Approved SQL string if all rules pass.
        
    Raises:
        Exception: With an error object if a rule violation is detected.
    """
    # Check for disallowed SQL patterns.
    for pattern in BUSINESS_RULES.get("disallowed_patterns", []):
        if re.search(pattern, sql, re.IGNORECASE):
            raise Exception({
                "errorCode": "ERR_CONSTRAINT_VIOLATION",
                "reason": f"SQL contains disallowed pattern: {pattern}",
                "hint": "Review your request and remove prohibited keywords."
            })
    # If it's a write operation and writes are globally disabled, reject.
    if query_type == "write" and not BUSINESS_RULES.get("allow_write_operations", True):
        raise Exception({
            "errorCode": "ERR_BUSINESS_RULE",
            "reason": "Write operations are disabled by policy.",
            "hint": "Use read operations or update the configuration."
        })
    op = extract_operation(sql)
    table = extract_table_name(sql, op)
    for rule in BUSINESS_RULES.get("rules", []):
        rule_op = rule.get("operation", "").upper()
        rule_table = rule.get("table", "").lower()
        if op == rule_op and table == rule_table:
            if not rule.get("allowed", True):
                error_message = rule.get("errorMessage", "Operation not allowed by business rules.")
                raise Exception({
                    "errorCode": "ERR_BUSINESS_RULE",
                    "reason": error_message,
                    "hint": "Review the business logic configuration."
                })
            # Detailed condition checks (if implemented) would go here.
    return sql



def execute_sql(sql):
    """
    Execute the SQL statement using SQLAlchemy.
    
    For SELECT statements, fetch and return the data.
    For INSERT/UPDATE/DELETE, commit the transaction and return the number of rows affected.
    """
    op = extract_operation(sql)
    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql))
            if op == "SELECT":
                rows = result.fetchall()
                result_list = [dict(row) for row in rows]
                return {"data": result_list}
            else:
                rowcount = result.rowcount
                return {"success": True, "rows_affected": rowcount}
    except (ProgrammingError, IntegrityError) as db_err:
        raise Exception({
            "errorCode": "ERR_CONSTRAINT_VIOLATION",
            "reason": "Database constraint or programming error encountered.",
            "hint": "Ensure your request meets the database schema requirements."
        })
    except SQLAlchemyError:
        raise Exception({
            "errorCode": "ERR_DATABASE",
            "reason": "An error occurred while executing the SQL statement.",
            "hint": "Check the SQL syntax and try again."
        })



def error_response(error_obj, status_code=400):
    """
    Return a JSON-formatted error response.
    """
    return jsonify({
        "errorCode": error_obj.get("errorCode", "UNKNOWN_ERROR"),
        "reason": error_obj.get("reason", "An error occurred."),
        "hint": error_obj.get("hint", "")
    }), status_code


# ------------------------------------------------------------------------------
# Response structures
# ------------------------------------------------------------------------------

class ResponseTableName(BaseModel):
    database: Optional[str]
    schema: Optional[str]
    table: Optional[str]

class ResponseOperation(BaseModel):
    operation: Literal['SELECT','DELETE','UPDATE']


class AgentResponseTableName(BaseModel):
    thoughts: str
    operation: ResponseOperation
    table_name: ResponseTableName

# ------------------------------------------------------------------------------
# Flask Endpoints
# ------------------------------------------------------------------------------

@app.route("/query", methods=["GET"])
async def query_endpoint():
    """
    GET /query for read-only operations.
    Expects a natural language query provided via the 'q' parameter or JSON body.
    """
    q = request.args.get("q")
    if not q and request.is_json:
        data = request.get_json()
        q = data.get("q")
    if not q:
        return error_response({
            "errorCode": "BAD_REQUEST",
            "reason": "Query parameter 'q' is required.",
            "hint": "Include your natural language query in the 'q' parameter."
        }, 400)
    try:
        sql = await parse_nl_to_sql(q)
        sql_meta = await parse_sql(sql.messages[1].content)
        sql_meta_formatted = AgentResponseTableName.model_validate_json(sql_meta.messages[-1].content)

    except ValueError as ve:
        return error_response({
            "errorCode": "ERR_PARSE_FAILURE",
            "reason": str(ve),
            "hint": "Rephrase your query using clearer language."
        }, 400)
    if sql_meta_formatted.operation != "SELECT":
        return error_response({
            "errorCode": "ERR_INVALID_ENDPOINT",
            "reason": "Non-SELECT operations are not allowed on the /query endpoint.",
            "hint": "Use the /mutate endpoint for data modifications."
        }, 400)
    try:

        approved_sql = apply_business_rules(sql.messages[1].content, query_type="read")
        result = execute_sql(approved_sql)
        return jsonify(result), 200
    except Exception as e:
        err_obj = e.args[0] if e.args else {}
        return error_response(err_obj, 400)



@app.route("/mutate", methods=["POST"])
async def mutate_endpoint():
    """
    POST /mutate for data mutation operations.
    Expects a JSON body with the field 'q' containing the natural language command.
    """
    if not request.is_json:
        return error_response({
            "errorCode": "BAD_REQUEST",
            "reason": "Request body must be JSON.",
            "hint": "Set the Content-Type header to application/json."
        }, 400)
    data = request.get_json()
    q = data.get("q")
    if not q:
        return error_response({
            "errorCode": "BAD_REQUEST",
            "reason": "Field 'q' is required in the JSON body.",
            "hint": "Provide a natural language command in the 'q' field."
        }, 400)
    try:
        sql = await parse_nl_to_sql(q)
        sql_meta = await parse_sql(sql.messages[1].content)
        sql_meta_formatted = AgentResponseTableName.model_validate_json(sql_meta.messages[-1].content)


    except ValueError as ve:
        return error_response({
            "errorCode": "ERR_PARSE_FAILURE",
            "reason": str(ve),
            "hint": "Rephrase your command to clearly indicate a data modification."
        }, 400)
    if sql_meta_formatted.operation == "SELECT":
        return error_response({
            "errorCode": "ERR_INVALID_ENDPOINT",
            "reason": "Read-only queries are not allowed on the /mutate endpoint.",
            "hint": "Use the /query endpoint for data retrieval."
        }, 400)
    try:
        approved_sql = apply_business_rules(sql.messages[1].content, query_type="write")
        result = execute_sql(approved_sql)
        return jsonify(result), 200
    except Exception as e:
        err_obj = e.args[0] if e.args else {}
        return error_response(err_obj, 400)

# ------------------------------------------------------------------------------
# Startup: Load Schema Information
# ------------------------------------------------------------------------------
with app.app_context():
    load_schema()

# ------------------------------------------------------------------------------
# Run the Flask App
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
