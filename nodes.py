import os
import json
import re
from pocketflow import Node, Flow

class TaskManager(Node):
    """
    任务经理节点:
    1. 指导科研文档每个部分的字数要求、内容细纲
    2. 生成配图描述（方法图和流程图）
    """
    def __init__(self, llm_client, prompt_file="prompts/task_manager_prompt.md", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_client = llm_client
        
        # 加载任务经理提示词
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
            
        # 确保基础输出目录存在
        os.makedirs("output", exist_ok=True)
            
    def prep(self, shared):
        """准备阶段：获取文档任务信息"""
        # 从shared中获取任务信息
        task_info = shared.get("task_info", {})
        return task_info
        
    def exec(self, task_info):
        """执行阶段：生成文档指南和配图描述"""
        # 构建提示词
        prompt = self.prompt_template.format(
            topic=task_info.get("topic", ""),
            style=task_info.get("style", "学术风格"),
            background=task_info.get("background", ""),
            method=task_info.get("method", ""),
            experiment=task_info.get("experiment", "")
        )
        
        # 调用LLM
        response = self.llm_client.generate(prompt)
        
        try:
            # 解析JSON响应
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # 如果不是有效的JSON，尝试提取JSON部分
            try:
                json_str = response.split("```json")[1].split("```")[0].strip()
                result = json.loads(json_str)
                return result
            except:
                # 失败时返回错误信息
                return {"error": "无法解析LLM响应为JSON格式", "raw_response": response}
    
    def post(self, shared, prep_res, exec_res):
        """后处理阶段：保存结果到shared"""

        if "error" in exec_res:
            print(f"错误: {exec_res['error']}")
            # 设置空的文档指南和配图描述，确保下游节点能继续工作
            shared["document_guide"] = {}
            shared["image_descriptions"] = []
            # return "default"  # 继续执行工作流
            return "error"
        
        # 根据任务主题创建输出目录（无论是否出错都需要创建）
        topic = shared["task_info"].get("topic", "untitled")
        # 将主题转换为有效的文件夹名
        folder_name = re.sub(r'[\\/*?:"<>|]', "_", topic)  # 替换不允许的字符
        folder_name = folder_name[:50] if len(folder_name) > 50 else folder_name  # 限制长度
        
        # 创建任务专属的输出目录
        output_dir = os.path.join("output", folder_name)
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # 保存输出路径到shared
        shared["output_dir"] = output_dir
        shared["images_dir"] = images_dir
        
            
        # 保存文档指南到shared
        shared["document_guide"] = exec_res.get("document_guide", {})
        
        # 保存配图描述到shared
        shared["image_descriptions"] = exec_res.get("image_descriptions", [])
        
        print(f"任务经理已完成分析，生成了文档指南和配图描述")
        print(f"为任务 '{topic}' 创建了输出目录: {output_dir}")
        
        # 返回默认动作，流转到下一个节点
        return "default"

class ArticleWriter(Node):
    """
    文章撰写者节点:
    根据任务经理提供的指南撰写文章
    """
    def __init__(self, llm_client, prompt_file="prompts/article_writer_prompt.md", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_client = llm_client
        
        # 加载文章撰写者提示词
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
    
    def prep(self, shared):
        """准备阶段：获取文档指南"""
        # 从shared中获取文档指南
        document_guide = shared.get("document_guide", {})
        task_info = shared.get("task_info", {})
        
        return {
            "document_guide": document_guide,
            "task_info": task_info
        }
    
    def exec(self, prep_data):
        """执行阶段：按照指南撰写文章"""
        document_guide = prep_data["document_guide"]
        task_info = prep_data["task_info"]
        
        # 构建提示词
        prompt = self.prompt_template.format(
            topic=task_info.get("topic", ""),
            style=task_info.get("style", "学术风格"),
            abstract_guide=json.dumps(document_guide.get("abstract", {}), ensure_ascii=False),
            background_guide=json.dumps(document_guide.get("background", ), ensure_ascii=False),
            method_guide=json.dumps(document_guide.get("method", {}), ensure_ascii=False),
            experiment_guide=json.dumps(document_guide.get("experiment", {}), ensure_ascii=False)
        )
        
        # 调用LLM生成文章
        response = self.llm_client.generate(prompt)
        
        try:
            # 尝试解析为JSON
            result = json.loads(response)
            return result
        except:
            # 如果不是JSON，返回原始文本
            return {"full_text": response}
    
    def post(self, shared, prep_res, exec_res):
        """后处理阶段：保存文章到文件"""
        # 提取文章内容
        if "full_text" in exec_res:
            article_content = exec_res["full_text"]
        else:
            # 组合各部分为完整文章
            sections = [
                exec_res.get("abstract", ""),
                exec_res.get("background", ""),
                exec_res.get("method", ""),
                exec_res.get("experiment", "")
            ]
            article_content = "\n\n".join([s for s in sections if s])
        
        # 找到第一个"#"开始的位置，截取后面的内容
        if "#" in article_content:
            first_hash_index = article_content.find("#")
            article_content = article_content[first_hash_index:]
        
        # 添加固定的开头文本
        fixed_header = '''# 特别注明
- 本文使用自建Agent工作流生成，包括文章和配图，github项目：[Jingye-Academic-Article-Generator](https://github.com/erliufashi/Jingye-Academic-Article-Generator)
- 写作部分使用gemini-2.5-pro-exp-03-25模型。
- 配图部分使用gpt-4o-image模型。

'''
        article_content = fixed_header + article_content
        
        # 将文章保存到shared
        shared["article_content"] = article_content
        
        # 获取输出目录路径
        output_dir = shared.get("output_dir", "output")
        
        # 将文章保存到文件
        document_path = os.path.join(output_dir, "document.md")
        with open(document_path, "w", encoding="utf-8") as f:
            f.write(article_content)
        
        print(f"文章已撰写完成并保存到 {document_path}")
        
        # 返回默认动作，流转到下一个节点
        return "default"

class ImageGenerator(Node):
    """
    配图生成者节点:
    根据任务经理提供的配图描述生成图像
    """
    def __init__(self, image_generator_client, prompt_file="prompts/image_generator_prompt.md", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_generator_client = image_generator_client
        
        # 加载配图生成者提示词
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        
    def prep(self, shared):
        """准备阶段：获取配图描述"""
        # 从shared中获取配图描述
        image_descriptions = shared.get("image_descriptions", [])
        # 获取图片输出路径
        images_dir = shared.get("images_dir", os.path.join("output", "images"))
        
        return {
            "image_descriptions": image_descriptions,
            "images_dir": images_dir
        }
    
    def exec(self, prep_data):
        """执行阶段：生成配图"""
        image_descriptions = prep_data["image_descriptions"]
        images_dir = prep_data["images_dir"]
        generated_images = []
        
        for desc in image_descriptions:
            # 获取图片索引和名称
            index = desc.get("index", 0)
            image_name = desc.get("name", f"图 {index}")
            
            # 构建图像生成提示词
            prompt = self.prompt_template.format(
                description=desc.get("description", ""),
                type=desc.get("type", "方法图"),
                keywords=", ".join(desc.get("keywords", []))
            )
            
            # 调用图像生成客户端
            image_path = os.path.join(images_dir, f"figure_{index}.png")
            
            try:
                # 生成图像并保存
                self.image_generator_client.generate_image(prompt, image_path)
                
                # 记录成功生成的图像信息
                generated_images.append({
                    "index": index,
                    "name": image_name,
                    "path": image_path,
                    "description": desc.get("description", ""),
                    "type": desc.get("type", "方法图"),
                    "caption": desc.get("caption", f"图 {index}: {image_name}")
                })
                
                print(f"已生成图像 {index}: {image_path}")
            except Exception as e:
                print(f"生成图像 {index} 失败: {str(e)}")
        
        return generated_images
    
    def post(self, shared, prep_res, exec_res):
        """后处理阶段：将图像信息加入shared"""
        # 保存生成的图像信息
        shared["generated_images"] = exec_res
        
        # 检查是否有错误发生（通过比较预期图像数量和实际生成的图像数量）
        expected_images = len(prep_res["image_descriptions"])
        actual_images = len(exec_res)
        
        if actual_images == expected_images:
            print(f"已成功生成 {len(exec_res)} 张图像")
        
        # 返回默认动作，流程结束
        return "default"
    
class EndNode(Node):
    """
    结束节点:
    结束工作流
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)