package com.congruens.rs

import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD

// TODO: replace by something with better performance than java's serializable
case class RatingDataSet(training: RDD[Rating], test: RDD[Rating]) extends java.io.Serializable
