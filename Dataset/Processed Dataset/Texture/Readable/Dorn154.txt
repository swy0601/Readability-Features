import unittest
import MySQLdb

class test_MySQLdb(dbapi20.DatabaseAPI20Test):
    driver = MySQLdb
    connect_args = ()
    connect_kw_args = dict(db='test',
                           read_default_file='~/.my.cnf',
                           charset='utf8',
                           sql_mode="ANSI,STRICT_TRANS_TABLES,TRADITIONAL")
