import os
import re
import json
from datetime import datetime
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, ProgrammingError

# ------------------------------------------------------------------------------
# Autogen Integration
# ------------------------------------------------------------------------------
try:
    from autogen import Agent
except ImportError:
    raise ImportError("Autogen module not found. Please install autogen to proceed.")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is required for Autogen integration.")

# Create an agent instance (model can be adjusted as needed)
autogen_agent = Agent(api_key=OPENAI_API_KEY, model="gpt-4")

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
BUSINESS_RULES = {
    "allow_write_operations": True,  # Global flag to allow/disallow mutations.
    "rules": [
        # Example: Block DELETE on the "users" table.
        {
            "operation": "DELETE",
            "table": "users",
            "allowed": False,
            "errorMessage": "Deletion of users is disallowed by policy."
        },
        # Example: Block specific UPDATE on "orders" when changing status from 'pending' to 'closed'.
        {
            "operation": "UPDATE",
            "table": "orders",
            "condition": "old_status = 'pending' AND new_status = 'closed'",
            "allowed": False,
            "errorMessage": "Cannot close an order if it is pending payment."
        }
    ],
    "disallowed_patterns": ["DROP", "ALTER", "TRUNCATE"]
}

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def extract_operation(sql):
    """
    Extract the SQL operation (first word) from the given SQL statement.
    """
    if not sql:
        return ""
    return sql.strip().split()[0].upper()

def extract_table_name(sql, operation):
    """
    Extract the target table name from the SQL statement using basic regex patterns.
    Supports SELECT, INSERT, UPDATE, and DELETE statements.
    """
    table = None
    if operation == "SELECT":
        match = re.search(r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE)
        if match:
            table = match.group(1)
    elif operation == "INSERT":
        match = re.search(r"INSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE)
        if match:
            table = match.group(1)
    elif operation == "UPDATE":
        match = re.search(r"UPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE)
        if match:
            table = match.group(1)
    elif operation == "DELETE":
        match = re.search(r"DELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE)
        if match:
            table = match.group(1)
    return table.lower() if table else None

def parse_nl_to_sql(nl_query, allowed_ops):
    """
    Use Autogen to convert a natural language query into a valid PostgreSQL SQL statement.
    
    Parameters:
        nl_query (str): The natural language request.
        allowed_ops (list): Allowed SQL operations (e.g. ["SELECT"] for queries,
                            or ["INSERT", "UPDATE", "DELETE"] for mutations).
                            
    Returns:
        sql (str): The generated SQL statement.
        
    Raises:
        ValueError: If the AI fails to parse the request or the generated operation is not allowed.
    """
    prompt = (
        "You are an AI assistant that converts natural language requests into valid PostgreSQL SQL queries.\n"
        f"Database schema: {json.dumps(SCHEMA_INFO)}\n"
        f"Allowed SQL operations: {', '.join(allowed_ops)}\n"
        f"User request: \"{nl_query}\"\n"
        "Respond only with a SQL query without any additional commentary."
    )
    try:
        response = autogen_agent.run(prompt)
        sql = response.strip()
    except Exception as e:
        raise ValueError("Autogen parsing failed: " + str(e))
    op = extract_operation(sql)
    if op not in allowed_ops:
        raise ValueError(f"Generated SQL operation '{op}' is not allowed. Expected one of {allowed_ops}.")
    return sql

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
# Flask Endpoints
# ------------------------------------------------------------------------------

@app.route("/query", methods=["GET"])
def query_endpoint():
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
        # Convert natural language to SQL, allowing only SELECT operations.
        sql = parse_nl_to_sql(q, allowed_ops=["SELECT"])
    except ValueError as ve:
        return error_response({
            "errorCode": "ERR_PARSE_FAILURE",
            "reason": str(ve),
            "hint": "Rephrase your query using clearer language."
        }, 400)
    if extract_operation(sql) != "SELECT":
        return error_response({
            "errorCode": "ERR_INVALID_ENDPOINT",
            "reason": "Non-SELECT operations are not allowed on the /query endpoint.",
            "hint": "Use the /mutate endpoint for data modifications."
        }, 400)
    try:
        approved_sql = apply_business_rules(sql, query_type="read")
        result = execute_sql(approved_sql)
        return jsonify(result), 200
    except Exception as e:
        err_obj = e.args[0] if e.args else {}
        return error_response(err_obj, 400)

@app.route("/mutate", methods=["POST"])
def mutate_endpoint():
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
        # Convert natural language to SQL, allowing only mutation operations.
        sql = parse_nl_to_sql(q, allowed_ops=["INSERT", "UPDATE", "DELETE"])
    except ValueError as ve:
        return error_response({
            "errorCode": "ERR_PARSE_FAILURE",
            "reason": str(ve),
            "hint": "Rephrase your command to clearly indicate a data modification."
        }, 400)
    if extract_operation(sql) == "SELECT":
        return error_response({
            "errorCode": "ERR_INVALID_ENDPOINT",
            "reason": "Read-only queries are not allowed on the /mutate endpoint.",
            "hint": "Use the /query endpoint for data retrieval."
        }, 400)
    try:
        approved_sql = apply_business_rules(sql, query_type="write")
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
