import com.amazonaws.auth.BasicAWSCredentials
import com.amazonaws.services.s3.AmazonS3Client
import com.amazonaws.services.s3.model.GetObjectRequest
import org.apache.spark
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}



object PageRank {

  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      println("Usage: PageRank InputDir OutputDir Iterations")
    }

    val sc = new SparkContext(new SparkConf().setAppName("PageRank").setMaster("local").set("spark.hadoop.validateOutputSpecs", "false"))

    sc.hadoopConfiguration.set("fs.s3a.access.key", "AKIAVE4VFGETRWHHZDGS")
    sc.hadoopConfiguration.set("fs.s3a.secret.key", "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")
    sc.hadoopConfiguration.set("fs.s3a.endpoint", "s3.amazonaws.com")


    val input_file = "s3a://bucket-bigdata-aws/airport.csv"
    val output_file = "s3a://bucket-bigdata-aws/rankresult"
    val iters = 10

//    val input_file = args(0)
//    val output_file = args(1)
//    val iters = args(2).toInt

    val lines=sc.textFile(input_file)

    val pairs = lines.map{ s =>
      val parts = s.split(",")
      (parts(0), parts(1))
    }


    val links = pairs.distinct().groupByKey()
    var ranks = links.mapValues(v => 1.0)


    for (i <- 1 to iters) {
      val contribs = links.join(ranks)
        .values
        .flatMap{ case (urls, rank) =>
          val size = urls.size
          urls.map(url => (url, rank / size))
        }
      ranks = contribs.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _)
    }

    val ranksResult= ranks.sortBy(-_._2)

    ranksResult.saveAsTextFile(output_file)

  }
}


