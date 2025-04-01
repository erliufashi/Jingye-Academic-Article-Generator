# -*- coding: utf-8 -*-
from pocketflow import Flow
from nodes import TaskManager, ArticleWriter, ImageGenerator, EndNode

from openai import OpenAI
import os
import json

# 实现OpenAI API的LLM客户端
class LLMClient:
    def __init__(self, api_key=None, model="gemini-2.5-pro-exp-03-25"):
        """
        初始化OpenAI客户端
        
        Args:
            api_key: OpenAI API密钥，默认从环境变量获取
            model: 使用的模型名称，默认为gpt-4-turbo
        """
        # 优先使用传入的api_key，否则从环境变量获取
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("需要提供OpenAI API密钥，可通过参数传入或设置OPENAI_API_KEY环境变量")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(base_url="https://xiaohumini.site/v1", api_key=self.api_key)
        # gemini-2.5 api暂时未开放，推荐使用中转站：https://xiaohumini.site/register?aff=wOJw

        self.model = model
    
    def generate(self, prompt):
        """
        调用OpenAI API生成文本
        
        Args:
            prompt: 提示词文本
            
        Returns:
            生成的文本响应
        """
        try:
            print("调用OpenAI API...")
            # 调用OpenAI的Chat Completion API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=8000
            )
            
            # 提取生成的内容
            content = response.choices[0].message.content
            print("API调用成功")
            return content
        except Exception as e:
            print(f"OpenAI API调用失败: {str(e)}")
            # 返回错误信息
            return json.dumps({
                "error": f"API调用失败: {str(e)}",
                "document_guide": {},
                "image_descriptions": []
            })

# 实现OpenAI DALL-E的图像生成客户端
class ImageGeneratorClient:
    def __init__(self, api_key=None, model="gpt-4o-image", size="1024x1024"):
        """
        初始化OpenAI DALL-E客户端
        
        Args:
            api_key: OpenAI API密钥，默认从环境变量获取
            model: 使用的模型名称，默认为dall-e-3
            size: 生成图像的尺寸，默认为1024x1024
        """
        # 优先使用传入的api_key，否则从环境变量获取
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供OpenAI API密钥，可通过参数传入或设置OPENAI_API_KEY环境变量")

        # 初始化OpenAI客户端
        self.client = OpenAI(base_url="https://xiaohumini.site/v1", api_key=self.api_key)
        # 推荐使用中转站：https://xiaohumini.site/register?aff=wOJw

        self.model = model
        self.size = size
    
    def generate_image(self, prompt, output_path):
        """
        调用OpenAI 4o-image生成图像并保存
        
        Args:
            prompt: 图像生成提示词
            output_path: 图像保存路径
        """
        try:
            print(f"调用4o-image API生成图像... 提示词: {prompt}...")
            
            # 调用OpenAI的图像生成API
            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                size=self.size,
                quality="standard",
                n=1
            )
            
            # 获取图像URL
            image_url = response.data[0].url
            
            # 下载图像并保存到指定路径
            import requests
            from PIL import Image
            from io import BytesIO
            
            # 下载图像
            image_response = requests.get(image_url)
            image = Image.open(BytesIO(image_response.content))
            
            # 保存图像
            image.save(output_path)
            print(f"图像已成功生成并保存至: {output_path}")
            
        except Exception as e:
            print(f"图像生成失败: {str(e)}")
            # 创建一个错误信息文本文件代替图像
            with open(output_path.replace(".png", ".txt"), "w", encoding="utf-8") as f:
                f.write(f"图像生成失败: {str(e)}\n提示词: {prompt}")
            print(f"错误信息已保存至: {output_path.replace('.png', '.txt')}")


def create_academic_workflow(llm_client, image_generator_client):
    """创建学术文档撰写工作流"""
    # 创建三个节点
    task_manager = TaskManager(llm_client, max_retries=3, wait=1)
    article_writer = ArticleWriter(llm_client)
    image_generator = ImageGenerator(image_generator_client)
    end_node = EndNode()

    # 连接节点
    task_manager >> article_writer >> image_generator
    task_manager - 'error' >> end_node
    
    # 创建工作流
    workflow = Flow(start=task_manager)
    
    return workflow

def run_workflow(task_info):
    """运行学术文档撰写工作流"""
    # 创建客户端
    llm_client = LLMClient()
    image_generator_client = ImageGeneratorClient()
    
    # 创建工作流
    workflow = create_academic_workflow(llm_client, image_generator_client)
    
    # 初始化共享数据
    shared = {"task_info": task_info}
    
    # 运行工作流
    print(f"开始为主题 '{task_info['topic']}' 生成学术文档...")
    workflow.run(shared)
    print("工作流程执行完成！")
    
    return shared

if __name__ == "__main__":
    # 示例任务信息
    example_task = {
"topic": "通过打篮球技术分析建立的飞机与鸡的群同态研究",
"background": '''
篮球运动，作为一项结合了高度技巧性、策略性与身体对抗性的全球性体育活动，其动作模式呈现出极其丰富的多样性。近年来，跨学科研究成为热点，本研究旨在探索看似无关领域间的深层结构关联。特别地，我们观察到两种极具代表性但风格迥异的篮球相关动作模式：一种以其超凡的滞空能力、高难度的后仰跳投和近乎偏执的精准度著称，其运动轨迹常被爱好者比拟为翱翔的飞行器，代表了篮球技艺的某种巅峰和传奇象征，虽然后期其运动生涯的终结方式带有令人扼腕的空中交通意外色彩；另一种则以其在练习场景中展现的独特律动、结合运球的标志性舞步而广为人知，因其特定发音和动作组合与某种家禽（特别是雄性）产生了奇妙的联想，并在网络文化中形成了现象级的传播。本研究首次尝试将这两种分别暗示“飞机”与“鸡”的篮球表现形式，纳入群论的数学框架，探究其内在结构是否可能存在同态映射关系。
''',
"method": '''
本研究采用多模态数据分析方法。
1. 数据采集：选取两类典型视频素材库。库A包含某已故传奇得分后卫大量比赛高光集锦，重点是其高难度投篮、空中对抗及展现“曼巴精神”的片段；库B包含某偶像艺人在练习生时期流出的著名篮球舞蹈片段及其后续模仿、二创视频。
2. 运动特征提取：利用计算机视觉和运动捕捉（模拟）技术，对两类素材中的关键动作进行量化。
* 对于“飞机”模式（记作群体Φ），提取指标包括：起跳高度、滞空时间、出手点稳定性、身体轴向控制精度、以及“关键球”情境下的动作选择概率。特别关注动作的突然性和不可预测性（尤其是在终结阶段）。
* 对于“鸡”模式（记作群体Γ），提取指标包括：运球节奏频率、肢体协调性（特别是肩膀与胯部的特定律动）、动作序列的重复性与变异性、以及“坤式”步伐的辨识度。
3. 群结构定义：
* 定义Φ的操作为“技术动作的组合与衔接”，如“突破-急停-后仰跳投”。
* 定义Γ的操作为“舞蹈/运球片段的序列化编排”，如“胯下运球-转身-特定步伐”。
4. 同态映射假设：构建一个映射函数 f: Φ → Γ，尝试寻找一个保持运算结构的映射关系。例如，是否可以将Φ中的“高强度对抗下的得分能力”映射到Γ中的“病毒式传播的模因强度”？是否可以将Φ中的“滞空控制”映射到Γ中的“原地律动稳定性”？
''',
"experiment": '''
1. 特征向量化与聚类：将提取的特征数据进行向量化表示，并使用K-means等聚类算法，在各自数据集中识别典型的动作“基元”。
2. 群运算模拟：基于定义的群操作，通过计算模拟动作序列的组合效果。例如，模拟“连续虚晃后的高难度上篮”（Φ）和“一套完整‘鸡你太美’舞步重复两次”（Γ）。
3. 同态验证：利用计算代数软件（如GAP）对假设的映射函数 f 进行检验。计算 f(a * b) 与 f(a) * f*(b) 之间的“结构距离”（例如，使用编辑距离或动态时间规整DTW）。统计显著性通过置换检验进行评估。
4. 预期结果：预期发现一种非平凡的弱同态。可能的结果包括：Φ中的“绝杀效率”与Γ中的“洗脑循环指数”存在某种正相关；Φ中动作的“飞行高度”与Γ中动作的“地面吸睛度”呈现反比关系。可能会观察到Φ数据集中存在少量与“意外失速/坠落”模式相关的异常动作轨迹，而在Γ数据集中则表现出极强的“自我复制”和“变异传播”特性。本研究或将揭示，无论是追求极致的空中技艺，还是引发地面狂欢的律动，其底层可能共享某种关于“影响力”或“表现力”的抽象结构。
'''
}
    
    # 运行工作流
    run_workflow(example_task)
