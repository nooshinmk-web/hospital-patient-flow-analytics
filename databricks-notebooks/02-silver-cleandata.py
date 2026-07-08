from pyspark.sql.types import *
from pyspark.sql.functions import *
from delta.tables import DeltaTable

# -------------------------
# ADLS Configuration 
# -------------------------
spark.conf.set(
    "fs.azure.account.key.<<storageaccount_name>>.dfs.core.windows.net",
   "<<storage_account_access_key>>"
)

spark.conf.set(
    "spark.databricks.delta.schema.autoMerge.enabled",
    "true"
)
bronze_path = "abfss://<<bronze_container>>@<<storageaccount_name>>.dfs.core.windows.net/<<file_path>>"
silver_path = "abfss://<<silver_container>>@<<storageaccount_name>>.dfs.core.windows.net/<<file_path>>"

# -------------------------
# Read from Bronze
# -------------------------
bronze_df =   ( 
    spark.readStream
  .format("delta")
  .load(bronze_path)
)

# -------------------------
# Define Schema
# -------------------------
schema = StructType([
  StructField("patient_id", StringType()),
  StructField("gender", StringType()),
  StructField("age", IntegerType()),
  StructField("department", StringType()),
  StructField("admission_time", StringType()),
  StructField("discharge_time", StringType()),
  StructField("bed_id", IntegerType()),
  StructField("hospital_id", IntegerType()),

# NEW ML / enrichment fields
  StructField("clinical_note", StringType()),
  StructField("stay_days", IntegerType()),
  StructField("intent_label", StringType())
])

# -------------------------
# Parse JSON to Dataframe
# -------------------------
parsed_df = bronze_df.withColumn("data", from_json(col("raw_json"), schema)).select("data.*")

# -------------------------
# 1. Schema Evolution
# -------------------------
expected_cols = ["patient_id","gender","age","department","admission_time","discharge_time","bed_id","hospital_id"]
for col_name in expected_cols:
   if col_name not in parsed_df.columns:
     clean_df = clean_df.withColumn(col_name, lit(None).cast(dtype))

# -------------------------
# 2. Data Cleansing
# -------------------------
# Type conversion
clean_df = parsed_df.withColumn(
     "admission_time", 
     to_timestamp("admission_time"))
clean_df = parsed_df.withColumn(
     "discharge_time", 
     to_timestamp("discharge_time"))
# Handle invalid timestamps
clean_df = clean_df.withColumn("admission_time", 
                                when(   
                                    (col("admission_time").isNull())| 
                                    (col("admission_time") > current_timestamp()), 
                                    current_timestamp()
                                ).otherwise(col("admission_time"))
                            )
# Handle Invalid Age
clean_df = clean_df.withColumn("age", 
                                when(   
                                 (col("age") < 0) | (col("age") > 100),
                                 floor(rand()*90+1).cast("int")
                              ).otherwise(col("age"))
                             )
# Deduplication
clean_df = clean_df.dropDuplicates(
    ["patient_id", "hospital_id", "admission_time"]
)

# -------------------------
# 3. Data Enrichment
# -------------------------
enriched_df = (
    clean_df
    .withColumn(
        "stay_days",
        datediff(col("discharge_time"), col("admission_time"))
    )
    .withColumn(
        "clinical_note",
       when(
        col("department") == "Cardiology",
        concat_ws(
            " ",
            expr("""
                CASE floor(rand()*3)
                    WHEN 0 THEN 'Patient admitted with chest discomfort and suspected cardiac involvement.'
                    WHEN 1 THEN 'Presentation consistent with possible cardiovascular symptoms requiring evaluation.'
                    ELSE 'Cardiac workup initiated due to reported chest-related symptoms.'
                END
            """),
            when(col("age") > 65,
                 expr("""
                    CASE floor(rand()*2)
                        WHEN 0 THEN 'Older patient with elevated cardiovascular risk requiring monitoring.'
                        ELSE 'Advanced age considered a risk factor for cardiac complications.'
                    END
                 """)
            ).otherwise(
                 expr("""
                    CASE floor(rand()*2)
                        WHEN 0 THEN 'Patient remains hemodynamically stable on admission.'
                        ELSE 'No acute cardiac instability observed during initial assessment.'
                    END
                 """)
            ),
            when(col("stay_days") > 5,
                 expr("""
                    CASE floor(rand()*2)
                        WHEN 0 THEN 'Prolonged hospitalization required for further cardiac evaluation.'
                        ELSE 'Extended monitoring performed due to persistent symptoms.'
                    END
                 """)
            ).otherwise(
                 expr("""
                    CASE floor(rand()*2)
                        WHEN 0 THEN 'Symptoms improved during hospital course.'
                        ELSE 'Clinical status remained stable throughout admission.'
                    END
                 """)
            )
        )
    )

    # -----------------------------
    # Neurology
    # -----------------------------
    .when(
        col("department") == "Neurology",
        concat_ws(
            " ",
            expr("""
                CASE floor(rand()*3)
                    WHEN 0 THEN 'Neurological evaluation performed due to headache and dizziness.'
                    WHEN 1 THEN 'Patient admitted for assessment of possible neurological disorder.'
                    ELSE 'Workup initiated for central nervous system symptoms.'
                END
            """),
            when(col("age") > 60,
                 expr("""
                    CASE floor(rand()*2)
                        WHEN 0 THEN 'Age-related neurological risk factors considered.'
                        ELSE 'Increased susceptibility to cerebrovascular conditions due to age.'
                    END
                 """)
            ).otherwise(
                 expr("""
                    CASE floor(rand()*2)
                        WHEN 0 THEN 'No prior neurological deficits reported.'
                        ELSE 'Baseline neurological status appears intact.'
                    END
                 """)
            ),
            expr("""
                CASE floor(rand()*2)
                    WHEN 0 THEN 'Follow-up neurological assessment recommended.'
                    ELSE 'Further monitoring advised for symptom progression.'
                END
            """)
        )
    )

    # -----------------------------
    # Orthopedics
    # -----------------------------
    .when(
        col("department") == "Orthopedics",
        concat_ws(
            " ",
            expr("""
                CASE floor(rand()*3)
                    WHEN 0 THEN 'Patient admitted for musculoskeletal evaluation.'
                    WHEN 1 THEN 'Orthopedic assessment performed due to mobility concerns.'
                    ELSE 'Evaluation of bone and joint condition initiated.'
                END
            """),
            when(col("age") > 70,
                 expr("""
                    CASE floor(rand()*2)
                        WHEN 0 THEN 'Increased fall risk associated with advanced age.'
                        ELSE 'Age-related mobility limitations observed.'
                    END
                 """)
            ).otherwise(
                 expr("""
                    CASE floor(rand()*2)
                        WHEN 0 THEN 'Mobility assessment performed.'
                        ELSE 'No major musculoskeletal abnormalities detected.'
                    END
                 """)
            ),
            expr("""
                CASE floor(rand()*2)
                    WHEN 0 THEN 'Rehabilitation plan initiated.'
                    ELSE 'Physical therapy recommended for recovery.'
                END
            """)
        )
    )

    # -----------------------------
    # Emergency
    # -----------------------------
    .when(
        col("department") == "Emergency",
        concat_ws(
            " ",
            expr("""
                CASE floor(rand()*3)
                    WHEN 0 THEN 'Patient presented to emergency department for acute evaluation.'
                    WHEN 1 THEN 'Emergency admission following acute clinical presentation.'
                    ELSE 'Urgent assessment conducted upon arrival at emergency unit.'
                END
            """),
            lit("Initial triage and stabilization completed."),
            when(col("stay_days") > 2,
                 expr("""
                    CASE floor(rand()*2)
                        WHEN 0 THEN 'Ongoing observation required after admission.'
                        ELSE 'Extended monitoring due to clinical condition.'
                    END
                 """)
            ).otherwise(
                 expr("""
                    CASE floor(rand()*2)
                        WHEN 0 THEN 'Patient stabilized and discharged after evaluation.'
                        ELSE 'Condition stabilized following initial treatment.'
                    END
                 """)
            )
        )
    )

    # -----------------------------
    # Default
    # -----------------------------
    .otherwise(
        lit("Patient admitted for clinical evaluation and routine medical management. Observations recorded during hospital stay.")
         ) 
    
     )
)
enriched_df = enriched_df.withColumn(
    "intent_label",
    when(col("department") == "Cardiology", lit("cardiac_case"))
    .when(col("department") == "Neurology", lit("neurology_case"))
    .when(col("department") == "Orthopedics", lit("orthopedic_case"))
    .when(col("department") == "Emergency", lit("emergency_case"))
    .otherwise(lit("general_case"))
)

# -------------------------
# 4. Text Cleaning
# -------------------------
silver_df = (
        enriched_df
        .dropna(subset=["clinical_note", "intent_label"])
        .withColumn(
            "clean_text",
            trim(lower(regexp_replace(col("clinical_note"), "[^a-zA-Z ]", ""))))
    )

# Validation
#df.select(count(when(col("clinical_note").isNull(), True)).alias("Null Notes"),count(when(col("intent_label").isNull(), True)).alias("Null Labels")).show()
#df.groupBy("intent_label").count().show()
#df_clean.show(100)

# -------------------------
# Write to Silver
# -------------------------
def upsert_to_silver(batch_df, batch_id):

    deltaTable = DeltaTable.forPath(spark, silver_path)

    (
        deltaTable.alias("t")
        .merge(
            batch_df.alias("s"),
            """
            t.patient_id = s.patient_id
            AND t.hospital_id = s.hospital_id
            AND t.admission_time = s.admission_time
            """
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

(silver_df.writeStream
    .foreachBatch(upsert_to_silver)
    .option("checkpointLocation", silver_path + "_checkpoint")
    .start()
)

