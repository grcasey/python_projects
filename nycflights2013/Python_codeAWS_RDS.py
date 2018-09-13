#Import libraries
import pandas as pd
import pymysql
import os

print(os.environ["MYSQL_HOST"])

#Connect to database
host = os.environ["MYSQL_HOST"]
port = int(os.environ["MYSQL_PORT"])
dbname = os.environ["MYSQL_DBNAME"]
user = os.environ["MYSQL_USER"]
password = os.environ["MYSQL_PASSWORD"]

conn = pymysql.connect(host, user=user, 
                       port=port,
                       passwd=password,
                       db=dbname)


# How many distinct origigns are there in the flights table?
pd.read_sql('SELECT COUNT(distinct origin) AS "Distinct Origins" FROM flights;', con=conn)

# How many flights were made on January first?
pd.read_sql('SELECT COUNT(year) FROM flights WHERE month=1 AND day=1;', con=conn)

# What's the most common carrier?
pd.read_sql('SELECT carrier, COUNT(*) AS "Total" FROM flights GROUP BY carrier ORDER BY Total desc limit 1;', con=conn)

# Which months have the highest number of flights?
%matplotlib inline
import matplotlib.pyplot as plt
pandas_df = pd.read_sql('SELECT month FROM flights;', con=conn)
pandas_df.groupby(pandas_df.month).size().plot(kind='bar', ylim=[24000,30000])
plt.xlabel('Month')
plt.ylabel('# of observation')
plt.show()

# What is the longest distance for each carrier?
pandas_df = pd.read_sql('SELECT carrier, distance FROM flights;', con=conn)
pandas_df.groupby(pandas_df.carrier).max().plot(kind='bar')
plt.xlabel('Carrier')
plt.ylabel('Longest distance')
plt.show()

# What are the five most common flight routes?
pandas_df = pd.read_sql('SELECT origin, dest, count(*) AS "Total" FROM flights GROUP BY origin, dest ORDER BY Total desc limit 5;', con=conn)
pandas_df

# Is the carrier with the highest departure delay also with the highest arrival delay?
highest_dep_delay = pd.read_sql('SELECT carrier, MAX(dep_delay) AS "Highest_dep_delay" FROM flights GROUP BY carrier ORDER BY Highest_dep_delay desc limit 1;', con=conn)
print(highest_dep_delay)

highest_arr_delay = pd.read_sql('SELECT carrier, MAX(arr_delay) AS Highest_arr_delay FROM flights GROUP BY carrier ORDER BY Highest_arr_delay desc limit 1;', con=conn)
print(highest_arr_delay)

# Are the unique destinations the same in all months?
pandas_df = pd.read_sql('SELECT DISTINCT month, dest FROM flights;', con=conn)
pandas_df.groupby(pandas_df.month).size().plot(kind='bar', ylim=[80,100])
plt.xlabel('Month')
plt.ylabel('# of observation')
plt.show()
