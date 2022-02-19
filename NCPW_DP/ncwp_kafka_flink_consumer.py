import os
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings

env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)
settings = EnvironmentSettings.new_instance().in_streaming_mode().use_blink_planner().build()
t_env = StreamTableEnvironment.create(env, environment_settings=settings)

kafka_jar_path = os.path.join(
  os.path.abspath(os.path.dirname(__file__)), "./",
  "flink-sql-connector-kafka_2.11-1.14.2.jar"
)
t_env.get_config().get_configuration().set_string(
  "pipeline.jars", f"file://{kafka_jar_path}"
)

souce_query = f"""
  create table source (
    states STRING,
    b_actions STRING,
    p_actions STRING
  ) with (
    'connector' = 'kafka',
    'topic' = 'NCPW-win',
    'properties.bootstrap.servers' = 'localhost:9091, localhost:9092, localhost:9093',
    'format' = 'json',
    'scan.startup.mode' = 'earliest-offset'
  )
"""

t_env.execute_sql(souce_query)

print('-------------')

sql = """
  SELECT p_actions FROM source
"""

res_table = t_env.sql_query(sql)
res_ds = t_env.to_data_stream(res_table)
res_ds.print()
env.execute()

# r1_sql = t_env.sql_query("""
#   SELECT states FROM source
# """)
#
# print('----start----')
#
# print(r1_sql.to_pandas())
#
# print('----end----')
#
# t_env.execute("ncwp_kafka_flink_consumer")