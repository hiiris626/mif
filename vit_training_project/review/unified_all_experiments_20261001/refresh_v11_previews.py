"""Extend the established 24-patch comparison with v10/v11, reuse old scores."""
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
old=P.parent/'v7_visualizations_20261001'
out=P.parent/'v11_visualizations_20261002'
out.mkdir(exist_ok=True)
for model in ['v1','v2','v3','v4','v5','v6','v7','v9','positive_dice_partial']:
    if not (out/model).exists():(out/model).symlink_to(old/model,target_is_directory=True)

source=(P/'refresh_v7_previews.py').read_text()
source=source.replace("OUT = P.parent / 'v7_visualizations_20261001'", "OUT = P.parent / 'v11_visualizations_20261002'")
source=source.replace("    spec.loader.exec_module(module)\n", "    spec.loader.exec_module(module)\n    if filename == 'evaluate.py':\n        for name in ['v10','v11']:\n            run = module.PROJECT/'experiments'/name/'runs'/name/'results'\n            module.MODELS[name] = ('classification', run/'model/best.pt', run/'thresholds.json')\n")
source=source.replace("'v7','v9','positive_dice_partial']", "'v7','v9','v10','v11','positive_dice_partial']")
source=source.replace("if '--render-only' in sys.argv:", "if '--render-only' in sys.argv or model not in ['v10','v11']:")
source=source.replace("(3,4),(16,13)", "(4,4),(16,16)")
source=source.replace("['H&E','GT dominant positive','v3','v5','v6','v7'],(2,3),(12,9)", "['H&E','GT dominant positive','v3','v6','v7','v9','v10','v11'],(2,4),(16,9)")
source=source.replace("['v5','v6','v7','v9']", "['v7','v10','v11']")
source=source.replace("GT / v5 / v6 / v7 / v9", "GT / v7 / v10 / v11")
exec(compile(source,str(P/'refresh_v7_previews.py'),'exec'))
