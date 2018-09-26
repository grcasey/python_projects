from scraper import scraper as sc
import json
import pymysql
import os
import csv

# connect to the cloud database via environment variables
# the application should be launched under bash-environment in order to pick up global variables
host = os.environ["MYSQL_HOST"]
port = int(os.environ["MYSQL_PORT"])
dbname = os.environ["MYSQL_DBNAME"]
user = os.environ["MYSQL_USER"]
password = os.environ["MYSQL_PASSWORD"]

db = pymysql.connect(host, 
                     user=user, 
                     port=port,
                     passwd=password,
                     db=dbname)

cursor = db.cursor()

# drop table for debug runs
# cursor.execute("""DROP TABLE property""")

# create a table in the DB
def create_table():
    sql = """CREATE TABLE nc_property(ID INT, ADDRESS VARCHAR(255), AREA_PROPERTY VARCHAR(255), BUILDING_TYPE VARCHAR(255), NEIGHBORHOOD VARCHAR(255), NUM_BATHROOM INT, NUM_FIREPLACE INT, OWNER1 VARCHAR(1000), OWNER2 VARCHAR(1000), TOTAL_VALUE VARCHAR(255), YEAR_BUILT INT, JDOC VARCHAR(2000), SALE_DATE DATE, HEAT VARCHAR(255), HEAT_FUEL VARCHAR(255), FOUNDATION VARCHAR(255), NUM_HALF_BATH INT, STORY VARCHAR(255), EXT_WALL VARCHAR(255), LAND_VALUE VARCHAR(255), BUILDING_VALUE VARCHAR(255), FEATURES_VALUE VARCHAR(255))"""
    cursor.execute(sql)

# write data fetched from the website to the DB
def write_to_db(search_results):
    for search_result in search_results:
        sql = """INSERT INTO nc_property(ID, ADDRESS, AREA_PROPERTY, BUILDING_TYPE, NEIGHBORHOOD, NUM_BATHROOM, NUM_FIREPLACE, OWNER1, OWNER2, TOTAL_VALUE, YEAR_BUILT, JDOC, SALE_DATE, HEAT, HEAT_FUEL, FOUNDATION, NUM_HALF_BATH, STORY, EXT_WALL, LAND_VALUE, BUILDING_VALUE, FEATURES_VALUE) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        val = (search_result["id"], search_result.get("address", None), search_result.get("area", None), search_result.get("building_type", None), search_result.get("neighborhood", None), search_result.get("num_bath", None), search_result.get("num_fireplaces", None), search_result.get("owner_1", None), search_result.get("owner_2", None), search_result.get("value_property", None), search_result.get("year_built", None), str(json.dumps(search_result.get("json_blob", None))), search_result.get("sale_date", None), search_result.get("heat", None), search_result.get("heat_fuel", None), search_result.get("foundation", None), search_result.get("num_half_bath", None), search_result.get("story", None), search_result.get("ext_wall", None), search_result.get("land_value", None), search_result.get("building_value", None), search_result.get("features_value", None))
        cursor.execute(sql, val)
        db.commit()

# return all rows from property table
def db_read_all():
    sql = """SELECT * FROM nc_property"""
    cursor.execute(sql)
    return cursor.fetchall()
    
# write data to CSV-file for further possibilities to access 
def write_to_csv_file(rows):
    file = open("input/property.csv", "w")
    csv_file = csv.writer(file)
    csv_file.writerows(rows)
    file.close()

# create table in db        
create_table()

# call the north caroline properties scraper
search_results = sc.search()

# write search result to db
write_to_db(search_results)
    
# read everything back from db
rows = db_read_all()    

# write it to csv file - will be use in modelling part
write_to_csv_file(rows)

db.close()