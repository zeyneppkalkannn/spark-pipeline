from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
)
from pyspark.sql.functions import col, regexp_replace, when

def get_schema():
    schema = StructType([
        StructField("DATETIME", TimestampType(), True),

        StructField("CELL", StringType(), True),
        StructField("CARRIER", IntegerType(), True),
        StructField("FREQ", DoubleType(), True),
        StructField("AZIMUTH", IntegerType(), True),

        StructField("LAT", DoubleType(), True),
        StructField("LON", DoubleType(), True),
        
        StructField("Traffic_Volume_Gbyte", DoubleType(), True),
        StructField("Nof_Avg_SimRRC_ConnUsr", DoubleType(), True),
        StructField("NOF_AVG_VOLTE_USER_HWI", DoubleType(), True),
        StructField("NUM_OF_RRC_Att", IntegerType(), True),
        StructField("ERAB_ESTAB_ATT", IntegerType(), True),

        StructField("AVGUL_RSSI_WEIGH_DBM_PUCCH", DoubleType(), True),
        StructField("AVGUL_RSSI_WEIGH_DBM_PUSCH", DoubleType(), True),
        StructField("AVG_RACH_TA", DoubleType(), True),
        StructField("VOLTE_TRAFFIC_ERL", DoubleType(), True),

        StructField("DLPRBUtilization", DoubleType(), True),
        StructField("DL_PRB_Util_%_HWI", DoubleType(), True),

        StructField("Avg_UL_RSRP_PUSCH", DoubleType(), True),
        StructField("Avg_UL_RSRP_PUCCH", DoubleType(), True),

        StructField("L.UL.Interference.Max", DoubleType(), True),
        StructField("L.UL.Interference.Avg", DoubleType(), True),
        StructField("L.UL.Interference.Min", DoubleType(), True),

        StructField("ERAB_ATTEMPT", IntegerType(), True),

        StructField("Average_UL_RSRP_PUSCH", DoubleType(), True),
        StructField("AvailTime_Auto_PC_HWI", DoubleType(), True),
        StructField("RRC_EstabSucc_PC", DoubleType(), True),
        StructField("ERAB_Drop_PC", DoubleType(), True),

        StructField("DL_PDCP_USER_THPUT", DoubleType(), True),
        StructField("UL_PDCP_USER_THPUT_HWI", DoubleType(), True),

        StructField("INTERFREQ_HO_SR", DoubleType(), True),
        StructField("INTRAFREQ_HO_SR", DoubleType(), True),

        StructField("ACTIVEUSER_DL_QCI1", DoubleType(), True),
        StructField("ACTIVEUSER_UL_QCI1", DoubleType(), True),

        StructField("Avg_CQI_HWI", DoubleType(), True),
        StructField("AVG_MCS_PUSCH", DoubleType(), True),
        StructField("AVG_SimRRC_ConnUsr_COUNT", DoubleType(), True),

        StructField("E-RAB_SETUP_SUCCESS_RATE", DoubleType(), True),
        StructField("HO_Succ_PC_In", DoubleType(), True),

        StructField("MAC_UL_IBLER", DoubleType(), True),
        StructField("MAC_DL_IBLER", DoubleType(), True),
        StructField("MAC_UL_RBLER", DoubleType(), True),
        StructField("MAC_DL_RBLER", DoubleType(), True),
        StructField("UL_PACKET_LOSS", DoubleType(), True),
        StructField("UL_PRB_Util_%", DoubleType(), True),
    ])

    for i in range(100):
        schema.add(StructField(f"L.UL.Interference.Avg.PRB{i}", DoubleType(), True))

    return schema

def get_overlap_schema():
    schema = StructType([
        StructField("CELL", StringType(), True),
        StructField("N_CELL", StringType(), True),
        StructField("Overlap_Alan%", DoubleType(), True),
    ])
    return schema