# 使用Miniconda作为基础镜像
FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /app

# 将environment.yml复制到容器中
COPY dl_environment.yml /app/dl_environment.yml

# 创建conda环境
RUN conda env create -f /app/dl_environment.yml

# 激活conda环境并设置环境变量
ENV PATH /opt/conda/envs/dl/bin:$PATH

# 将宿主机的当前目录挂载到容器中
VOLUME /app

# 定义启动命令
# 注意：将your_script.py替换为你想要运行的脚本
# CMD ["python", "your_script.py"]



# docker build -t Manifold_SmiLearn .
# docker run -it --rm -v $(pwd):/app Manifold_SmiLearn
