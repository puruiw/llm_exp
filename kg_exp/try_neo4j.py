from neo4j import GraphDatabase, exceptions

uri = "bolt://localhost:7687"
auth = ("neo4j", "password")
driver = GraphDatabase.driver(uri, auth=auth)
try:
    driver.verify_connectivity()
    print("driver connected")
except exceptions.AuthError:
    print("auth failed")
except Exception as e:
    print("other error", e)
finally:
    driver.close()
