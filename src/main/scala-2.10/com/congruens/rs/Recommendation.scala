package com.congruens.rs

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, ALS, Rating}

import util.Random

object Recommendation {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    if (args.length < 2) {
      System.err.println("Usage: Recommendation <host> <file_path>")
      System.exit(1)
    }
    val host = args(0)
    val filePath = args(1)

    val conf = new SparkConf().setAppName("Recommendation").setMaster(host)
    val sc = new SparkContext(conf)

    val feedbackData = sc.textFile(filePath)

    def separateDataSet(dataSet: RDD[Rating]): RatingDataSet = {
      val dataSetWithRandomNumber = dataSet.map(rating => (rating, Random.nextInt(10)))
      val deleteRandomNumber = (ratingWithRandomNumber: (Rating, Int)) => ratingWithRandomNumber._1

      def filterByRandomNumber(filteringFunction: ((Rating, Int)) => Boolean) =
        dataSetWithRandomNumber
          .filter(filteringFunction)
          .map(deleteRandomNumber)
          .cache()

      new RatingDataSet(
        training = filterByRandomNumber(e => e._2 < 9),
        test = filterByRandomNumber(e => e._2 >= 9)
      )
    }

    // view count mapping: if never view or view once then 0 else > 0
    def ratingMapping(rate: Double): Double = 1 - 1 / (1 + rate)

    val ratings = feedbackData.map(_.split("::") match { case Array(user, item, rate) =>
        Rating(user.toInt, item.toInt, ratingMapping(rate.toDouble))
    })

    val dataSet = separateDataSet(ratings)

    println(
      "training: " + dataSet.training.count +
      ", test: " + dataSet.test.count
    )

    val ranks = List(20) // the bigger, the better
    val lambdas = List(1.2, 2.0, 3.4)
    val numIters = List(22) // the bigger, the better
    val alphas = List(500.0)

    val selectedModel = findBestModel(dataSet, ranks, numIters, lambdas, alphas)

    val ndcgResult: Double = selectedModel match {
      case Some(m) => computeNdcg(m, dataSet.test, 10)
      case None => 0.0
    }
    println("The best model got ndcg at 10: " + ndcgResult)

    sc.stop()
  }

  def findBestModel(
      feedbackData: RatingDataSet,
      ranks: List[Int],
      numIters: List[Int],
      lambdas: List[Double],
      alphas: List[Double]): Option[MatrixFactorizationModel] = {

    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationNdcg = Double.MinValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for (rank <- ranks; lambda <- lambdas; alpha <- alphas; numIter <- numIters) {
      val model = ALS.trainImplicit(feedbackData.training, rank, numIter, lambda, alpha)
      val validationNdcg = computeNdcg(model, feedbackData.test, 10)
      println("NDCG (validation) = " + validationNdcg + " for the model trained with rank = "
        + rank + ", lambda = " + lambda + ", alpha = " + alpha + ", and numIter = " + numIter + ".")

      if (validationNdcg > bestValidationNdcg) {
        bestModel = Some(model)
        bestValidationNdcg = validationNdcg
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    bestModel
  }

  def computeNdcg(model: MatrixFactorizationModel, data: RDD[Rating], k: Int): Double = {
    val binarizedRatings = data.map(r =>
        Rating(r.user, r.product, if (r.rating > 0) 1.0 else 0.0)).cache()
    val userRecommended = model.recommendProductsForUsers(k)
    val userItems = binarizedRatings.groupBy(_.user)

    val relevantDocuments = userItems.join(userRecommended).map {
      case (user, (actual, predictions)) =>
        (predictions.map(_.product), actual.filter(_.rating > 0.0).map(_.product).toArray)
    }

    new RankingMetrics(relevantDocuments).ndcgAt(k)
  }
}
