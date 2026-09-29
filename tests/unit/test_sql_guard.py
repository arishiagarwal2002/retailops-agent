import pytest

from tools.sql import GuardrailError, validate_sql


@pytest.mark.parametrize("query", [
    "SELECT 1; DROP TABLE products",
    "DELETE FROM transactions",
    "UPDATE products SET category = 'x'",
    "INSERT INTO products VALUES ('x')",
    "DROP TABLE customers",
    "TRUNCATE transactions",
    "WITH x AS (DELETE FROM transactions RETURNING *) SELECT * FROM x",
    "SELECT * INTO backup FROM products",
    "SELECT * FROM products FOR UPDATE",
    "SELECT * FROM runs",
    "SELECT * FROM api_keys",
    "SELECT * FROM pg_catalog.pg_roles",
    "SELECT * FROM information_schema.tables",
    "SELECT pg_sleep(10)",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT set_config('statement_timeout', '0', false)",
    "COPY products TO STDOUT",
    "GRANT ALL ON products TO public",
    "",
    "SELEC * FRM products",
])
def test_rejects_unsafe_or_invalid(query):
    with pytest.raises(GuardrailError):
        validate_sql(query)


@pytest.mark.parametrize("query", [
    "SELECT * FROM products",
    "select stock_code, sum(quantity) from transactions group by 1",
    "WITH t AS (SELECT stock_code FROM products) SELECT * FROM t",
    "SELECT stock_code FROM products UNION SELECT stock_code FROM stock_levels",
    "SELECT * FROM (SELECT stock_code FROM products) AS sub",
    "SELECT p.stock_code FROM products p JOIN stock_levels s USING (stock_code)",
    "SELECT * FROM products;",
])
def test_allows_read_queries(query):
    assert validate_sql(query)


def test_limit_added_when_missing():
    assert validate_sql("SELECT * FROM products", row_limit=1000).endswith("LIMIT 1000")


def test_limit_capped_when_too_large():
    assert "LIMIT 1000" in validate_sql("SELECT * FROM products LIMIT 50000", row_limit=1000)


def test_small_limit_kept():
    assert "LIMIT 5" in validate_sql("SELECT * FROM products LIMIT 5", row_limit=1000)


def test_error_message_names_allowed_tables():
    with pytest.raises(GuardrailError, match="Allowed tables"):
        validate_sql("SELECT * FROM pending_actions")
