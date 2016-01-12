package com.congruens.rs

import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD

case class RatingDataSet(training: RDD[Rating], validation: RDD[Rating], test: RDD[Rating])
