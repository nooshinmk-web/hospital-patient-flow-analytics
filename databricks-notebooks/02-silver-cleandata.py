from pyspark.sql.types import *
from pyspark.sql.functions import *

#ADLS configuration 
spark.conf.set(
  "fs.azure.account.key.<<Storageaccount_name>>.dfs.core.windows.net",
  "<<Storage_Account_access_key>>"
)
bronze_path = "abfss://<<container>>@<<Storageaccount_name>>.core.windows.net/<<path>>"
silver_path = "abfss://<<container>>@<<Storageaccount_name>>.core.windows.net/<<path>>"

#read from bronze
bronze_df =   ( 
    spark.readStream
  .format("delta")
  .load(bronze_path)
)
#silver_df = bronze_df.select("id","name","age","salary")
#silver_df.writeStream.format("delta").option("checkpointLocation", "abfss://silver@hospitalpatientstorage.dfs.core.windows.net/p
#Define Schema
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
 
 #parse it to dataframe
parsed_df = bronze_df.withColumn("data", from_json(col("raw_json"), schema)).select("data.*")
 #convert type to Timestamp
clean_df = parsed_df.withColumn(
     "admission_time", 
     to_timestamp("admission_time"))
clean_df = parsed_df.withColumn(
     "discharge_time", 
     to_timestamp("discharge_time"))
 #invalid admission_time
clean_df = clean_df.withColumn("admission_time", 
                                when(   
                                (col("admission_time").isNull())| (col("admission_time") > current_timestamp())
                                , current_timestamp()
                                ).otherwise(col("admission_time"))
                                )
 #Handle Invalid Age
clean_df = clean_df.withColumn("age", 
                                when(   
                                (col("age") < 0) | (col("age") > 100),
                                floor(rand()*90+1).cast("int")
                                ).otherwise(col("age"))
                                )
 #Schema evolution
expected_cols = ["patient_id","gender","age","department","admission_time","discharge_time","bed_id","hospital_id"]
for col_name in expected_cols:
   if col_name not in clean_df.columns:
     clean_df = clean_df.withColumn(col_name, lit(None).cast(dtype))
# -------------------------
# ENRICHMENT STEP
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
 #write to silver
(enriched_df.writeStream
  .format("delta")
  .outputMode("append")
  .option("mergeSchema", "true")
  .option("checkpointLocation", silver_path + "_checkpoint")
  .start(silver_path)
 )
