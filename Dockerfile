# --- Base Image ---
FROM openjdk:11-jre-slim

# --- Environment Variables ---
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${PATH}"
ENV PYTHONPATH="${SPARK_HOME}/python/:${SPARK_HOME}/python/lib/py4j-0.10.9.7-src.zip"
ENV PYSPARK_PYTHON=python3

# --- Install Dependencies ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        wget \
        bash \
        tini && \
    rm -rf /var/lib/apt/lists/*

# --- Install Spark ---
RUN wget -q "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" && \
    tar -xzf "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -C /opt/ && \
    mv "/opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" "${SPARK_HOME}" && \
    rm "spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"

# --- Install Jupyter ---
RUN pip3 install notebook jupyterlab

# --- Expose port for Jupyter ---
EXPOSE 8888

# --- Start Jupyter automatically ---
CMD ["tini", "--", "python3", "-m", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
