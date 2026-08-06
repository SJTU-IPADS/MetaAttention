import jinja2
import os
import os.path as osp


TEMPLATE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), 'cute_template')
OUTPUT_DIR = osp.join(
    osp.dirname(
        osp.abspath(__file__)),
    'cute_template_output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# with open('/home/aiscuser/cfy/flash-attention-hopper-template/template/online_func.h', 'r') as f:
#     cute_template = f.read()
# CUTE_ONLINE_FUNC = cute_template


class CuteAttnTemplate:
    def __init__(self, template_dir=TEMPLATE_DIR, output_dir=OUTPUT_DIR,
                 **kwargs):
        
        for root, dirs, files in os.walk(template_dir):
            # 计算相对于模板目录的相对路径
            rel_path = os.path.relpath(root, template_dir)
            
            # 在输出目录中创建对应的子目录
            dest_dir = os.path.join(output_dir, rel_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    template1 = f.read()
                
                self.render_code(template1, **kwargs)
                
                # 在输出目录的对应子目录中创建文件
                dest_file = os.path.join(dest_dir, file)
                
                # 检查文件是否已存在且内容相同
                skip_write = False
                if os.path.exists(dest_file):
                    with open(dest_file, 'r') as f:
                        existing_content = f.read()
                    if existing_content == self.tlcode:
                        skip_write = True
                
                # 只有当文件不存在或内容不同时才写入
                if not skip_write:
                    with open(dest_file, 'w') as f:
                        f.write(self.tlcode)

    def render_code(self, temp_code, **kwargs):
        template = jinja2.Template(
            temp_code
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.tlcode = template.render(**kwargs)

    def __call__(self):
        return self.tlcode
