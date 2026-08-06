import jinja2
import os
import os.path as osp

TEMPLATE_DIR = osp.join(
    osp.dirname(
        osp.abspath(__file__)),
    'tl_template/attn/blockattn_tl.py')


class TlBlockAttnTemplate:
    def __init__(self, template_dir=TEMPLATE_DIR,
                 **kargs
                 ):
        with open(template_dir, 'r') as f:
            TL_KERNEL = f.read()

        template = jinja2.Template(TL_KERNEL)

        # remove None
        kargs = {k: (v if v is not None else "") for k, v in kargs.items()}
        self.tlcode = template.render(**kargs)

    def __call__(self):
        return self.tlcode


