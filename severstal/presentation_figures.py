"""Redraw presentation evidence from the existing archive; no inference or training.

Run from the repository root with the lightweight requirements-figures environment.
"""

import argparse
import csv
import html
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/severstal-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch
import numpy as np
from PIL import Image


COLORS = ["#228b70", "#dc822d", "#5176b6", "#ac5298"]
COMBOS = ["A", "B", "C", "AB", "AC", "BC", "ABC"]
BRANCHES = ["raw", "area", "fixed5", "hard", "soft", "dynamic"]
BRANCH_LABELS = ["原始", "轻面积", "5倍面积", "硬门控", "概率相乘", "动态面积"]
RUNS = {"A": "E02-r01", "B": "E02-r03", "C": "E02-r04"}
FIXED = ["8ed6b8054", "8f81e6d35", "3088a6a0d", "7f3181e44"]
EXAMPLES = ["f9b98ab64", "163ff3547", "6668e50d0", "9699f82fe", "0074d81d0", "07cbfd1f8"]


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def decode(encoded, height=256, width=1600):
    """Same column-major, 1-index convention as data.decode_rle, without torch."""
    flat = np.zeros(height * width, dtype=np.uint8)
    runs = np.fromstring(encoded, sep=" ", dtype=np.int64).reshape(-1, 2)
    for start, length in runs:
        if start < 1 or length < 0 or start - 1 + length > flat.size:
            raise ValueError("Invalid RLE")
        flat[start - 1:start - 1 + length] = 1
    if flat.sum() != runs[:, 1].sum():
        raise ValueError("Overlapping RLE intervals")
    return flat.reshape((height, width), order="F")


def overlay(image, masks):
    out = image.astype(float) / 255
    for i, mask in enumerate(masks):
        color = matplotlib.colors.to_rgb(COLORS[i])
        out[mask.astype(bool)] = 0.45 * out[mask.astype(bool)] + 0.55 * np.array(color)
    return out


class EvidenceFigures:
    def __init__(self, raw, output):
        self.raw, self.output = Path(raw), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
        if font.exists():
            font_manager.fontManager.addfont(font)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
        else:
            plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
        plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.unicode_minus": False,
                             "figure.facecolor": "white", "savefig.facecolor": "white"})
        self.cards = []
        self.audit = {"basis": "2026-10-24 existing evidence redraw; no new inference or training", "raw_root": str(self.raw), "source_commit": "50fbe58", "source_commit_scope": "repository before visualization edits; training commits are in E02/E03 records"}
        self.split = read_json(self.raw / "data/split_seed42.json")
        self.names = self.split["train"] + self.split["val"]
        assert len(self.names) == len(set(self.names)) == 12568
        self.labels = {name: ["", "", "", ""] for name in self.names}
        self.areas = {name: np.zeros(4, dtype=int) for name in self.names}
        with (self.raw / "data/severstal/train.csv").open() as handle:
            for row in csv.DictReader(handle):
                name, c, encoded = row["ImageId"], int(row["ClassId"]) - 1, row["EncodedPixels"]
                self.labels[name][c] = encoded
                self.areas[name][c] = np.fromstring(encoded, sep=" ", dtype=np.int64)[1::2].sum()
        self.summary = read_json("docs/results/E03-r02/summary.json")
        self.comparison = read_json("docs/results/E03-r02/class2_positive_comparison.json")
        self.histories = {m: read_rows(self.raw / f"runs/{r}/history.jsonl") for m, r in RUNS.items()}

    def image_masks(self, stem):
        image = np.asarray(Image.open(self.raw / f"data/severstal/train_images/{stem}.jpg").convert("RGB"))
        masks = np.stack([decode(rle) for rle in self.labels[stem + ".jpg"]])
        assert np.array_equal(masks.sum(axis=(1, 2)), self.areas[stem + ".jpg"])
        return image, masks

    def save(self, fig, slug, title, sources, observation, scope, footer="已有记录重绘 · 2026-10-24"):
        fig.suptitle(title, fontsize=21, fontweight="bold", y=0.985)
        fig.text(0.02, 0.015, footer, fontsize=10, color="#555555")
        for ext in ("png", "svg"):
            fig.savefig(self.output / f"{slug}.{ext}", dpi=190, bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)
        sources = [('docs/' + s[6:]) if s.startswith('../../') else ('docs/results/' + s[3:]) if s.startswith('../') else s for s in sources]
        self.cards.append(dict(id=slug, title=title, sources=sources, observation=observation, scope=scope))

    def dataset(self):
        areas = np.stack(list(self.areas.values()))
        presence = (areas > 0).astype(np.int64)
        positive_counts = presence.sum(0)
        count_defect = int(presence.any(1).sum())
        count_multi = int((presence.sum(1) > 1).sum())
        assert positive_counts.tolist() == [897, 247, 5150, 801]
        assert (count_defect, count_multi) == (6666, 427)
        self.audit["dataset"] = dict(images=len(areas), positive_images=count_defect,
                                     normal_images=len(areas)-count_defect, multi_class=count_multi,
                                     positive_counts=positive_counts.tolist(), cooccurrence=(presence.T @ presence).tolist())
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.subplots_adjust(top=.89, bottom=.1, hspace=.47, wspace=.25)
        ax = axs[0, 0]
        bars = ax.bar(["类别 1", "类别 2", "类别 3", "类别 4"], positive_counts, color=COLORS)
        ax.bar_label(bars, padding=4); ax.set_ylim(0, 5900); ax.set_title("阳性原图数：类别 2 最少"); ax.set_ylabel("原图数")
        ax = axs[0, 1]
        bars = ax.barh(["正常原图", "至少一类缺陷", "其中：多类共存"], [5902, count_defect, count_multi], color=["#999999", COLORS[0], COLORS[2]])
        ax.bar_label(bars, padding=5); ax.set_xlim(0, 8000); ax.set_title("第三项包含在第二项中"); ax.invert_yaxis()
        ax = axs[1, 0]
        matrix = presence.T @ presence
        im = ax.imshow(matrix, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i,j]), ha="center", va="center", color="white" if matrix[i,j] > 2600 else "black")
        ax.set_xticks(range(4), ["类1", "类2", "类3", "类4"]); ax.set_yticks(range(4), ["类1", "类2", "类3", "类4"])
        ax.set_title("类别共现：同图可以有多个标签"); fig.colorbar(im, ax=ax, shrink=.75, label="原图数")
        ax = axs[1, 1]
        ax.boxplot([areas[areas[:, c] > 0, c] for c in range(4)], tick_labels=["类1", "类2", "类3", "类4"], showfliers=False)
        ax.set_yscale("log"); ax.set_ylabel("真实掩膜面积 / 像素（对数轴）"); ax.set_title("阳性面积分布：箱体为中间 50%，省略离群点")
        self.save(fig, "01_dataset", "从数据出发：类别不平衡、正常图与多类共存", ["raw/data/severstal/train.csv", "raw/data/split_seed42.json"],
                  "12,568 张原图中，5,902 张正常；类别 2 仅 247 张阳性。四类数量不能相加当作原图总数。", "统计对象为有标签的全部训练数据；测试集没有真值，不参与此图。")
        fig, axs = plt.subplots(6, 2, figsize=(17, 12))
        fig.subplots_adjust(top=.92, bottom=.07, hspace=.38, wspace=.04)
        for row, stem in enumerate(EXAMPLES):
            image, masks = self.image_masks(stem)
            classes = [str(i+1) for i in range(4) if masks[i].any()]
            label = "类别 " + "+".join(classes) if classes else "正常：四类真值均空"
            axs[row,0].imshow(image); axs[row,1].imshow(overlay(image, masks))
            axs[row,0].set_title(f"{stem}.jpg · {label}", loc="left", fontsize=11)
            axs[row,1].set_title("原图叠加真实掩膜 · 面积 " + "/".join(str(int(x.sum())) for x in masks), loc="left", fontsize=11)
            for ax in axs[row]: ax.axis("off")
        fig.legend(handles=[Patch(color=color,label=f"类别 {i+1}") for i,color in enumerate(COLORS)], loc="lower right", ncol=4, bbox_to_anchor=(.97,.015))
        self.save(fig, "02_examples", "原始图像与标注：四类、正常图、多类图", ["raw/data/severstal/train_images/", "raw/data/severstal/train.csv", "../../答辩材料索引.md"],
                  "缺陷有不同尺度和外观；正常图也可能有黑区和规则纹理。彩色区域是真实标签，不是模型预测。", "四类各取验证集面积排序中间样本，正常与多类各取文件名排序首图；不能代表所有形态。")

    def example_details(self):
        fig,axs=plt.subplots(2,4,figsize=(16,9))
        fig.subplots_adjust(top=.86,bottom=.12,hspace=.2,wspace=.08)
        crops=[]
        for c,stem in enumerate(EXAMPLES[:4]):
            image,masks=self.image_masks(stem)
            yy,xx=np.where(masks[c])
            closest=int(np.argmin((yy-yy.mean())**2+(xx-xx.mean())**2))
            left=int(np.clip(xx[closest]-128,0,1600-256))
            im=image[:,left:left+256];ms=masks[:,:,left:left+256]
            axs[0,c].imshow(im);axs[1,c].imshow(overlay(im,ms))
            axs[0,c].set_title(f'类别 {c+1} · {stem}',fontsize=12)
            fraction=float(ms[c].sum()/masks[c].sum())
            axs[1,c].set_title(f'真值叠加 · 保留本类真值面积 {fraction:.1%}',fontsize=11)
            crops.append(dict(image=stem+'.jpg',class_id=c+1,left=left,width=256,area_fraction=fraction))
        for ax in axs.flat:ax.axis('off')
        self.audit['example_detail_crops']=crops
        self.save(fig,'02b_example_details','局部放大：把细窄缺陷看清楚',['raw/data/severstal/train_images/','raw/data/severstal/train.csv'],
                  '同一批四类原图的局部视图，颜色仅表示真实标签。细窄标签在全图缩小时容易看不清。',
                  '横向以距本类掩膜重心最近的真值像素为中心，截取256×256窗口；保留全部高度。图下注明窗口内本类面积占比；不是模型预测。')

    def processing(self):
        fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        nodes = [(.03,.7,.23,.18,"图像目录 + RLE 标注\n补齐 CSV 中缺失的空标签\n12,568 张原图"),
                 (.36,.7,.26,.18,"先按原图划分 · seed 42\n多标签分层，包含正常图\n训练 10,054 / 验证 2,514"),
                 (.04,.39,.27,.18,"分割训练\n256×512 随机横向裁剪\n图像和掩膜同步翻转"),
                 (.36,.39,.26,.18,"分类训练\n完整 256×1600 图像\n四类独立存在性标签"),
                 (.69,.39,.28,.18,"验证与选模\n完整 256×1600 图像\n分割：非空 Dice\n分类：macro-F1"),
                 (.07,.08,.38,.18,"各成员 sigmoid 概率 → 等权平均\n比较七组合 × 六种处理\n记录得分、定位、误报与逐图变化"),
                 (.57,.08,.37,.18,"冻结 area / soft 两分支\n官方环境推理 → 列优先 RLE\n记录 Public / Private 得分")]
        for x,y,w,h,label in nodes:
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012",facecolor="#edf3f8",edgecolor="#7c99ae"))
            ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=13)
        for start,end in [((.26,.79),(.35,.79)),((.48,.69),(.18,.58)),((.49,.69),(.49,.58)),((.61,.74),(.82,.58)),((.82,.38),(.44,.25)),((.45,.17),(.56,.17))]:
            ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color="#465869",lw=2))
        ax.annotate("",xy=(.69,.56),xytext=(.31,.56),arrowprops=dict(arrowstyle="->",color="#465869",lw=2,connectionstyle="arc3,rad=-.35"))
        ax.annotate("",xy=(.69,.48),xytext=(.62,.48),arrowprops=dict(arrowstyle="->",color="#465869",lw=2))
        ax.text(.5,.97,"RGB /255，再按 ImageNet 均值与标准差归一化；亮度增强只改图像，不改标签",ha="center",fontsize=12)
        self.save(fig, "03_data_flow", "数据如何进入训练、融合和官方测试", ["severstal/data.py", "configs/", "../../决策记录.md#d33", "../../决策记录.md#d49"],
                  "先划分原图再裁剪，避免同一图像的不同裁块跨训练与验证；分类器使用全图。", "按现有代码与配置绘制的流程示意，不是新增实验。")
        stem = "163ff3547"; image, masks = self.image_masks(stem)
        col = masks[1].sum(0); sums = np.convolve(col, np.ones(512,dtype=int), mode="valid")
        keep = int(np.median(np.flatnonzero(sums == sums.max()))); empty = int(np.flatnonzero(sums == 0)[0])
        crop, cmask = image[:, keep:keep+512], masks[:,:,keep:keep+512]
        fig = plt.figure(figsize=(16,8)); grid = fig.add_gridspec(2,4,height_ratios=[1,1.4],hspace=.42)
        fig.subplots_adjust(top=.88,bottom=.14)
        ax = fig.add_subplot(grid[0,:]); ax.imshow(overlay(image,masks)); ax.axis("off")
        for left,color,label in [(keep,COLORS[0],"保留缺陷的窗口"),(empty,COLORS[1],"没有缺陷的窗口")]:
            ax.add_patch(Rectangle((left,0),512,255,fill=False,edgecolor=color,lw=3)); ax.text(left+5,18,label,color="white",fontsize=12,bbox=dict(facecolor=color,alpha=.9,pad=3))
        variants = [(crop,cmask,"裁剪：图像与掩膜同步"),(image[:,empty:empty+512],masks[:,:,empty:empty+512],"另一窗口：本类标签变空"),(crop[:,::-1],cmask[:,:,::-1],"水平翻转：标签同步"),(np.clip(crop[:,::-1].astype(float)*1.1+15,0,255).astype(np.uint8),cmask[:,:,::-1],"亮度/对比度：标签不变")]
        for i,(im,mask,title) in enumerate(variants):
            ax=fig.add_subplot(grid[1,i]); ax.imshow(overlay(im,mask));ax.set_title(title,fontsize=12);ax.axis("off")
        self.audit["illustration_crop"] = dict(image=stem+".jpg",keep_left=keep,empty_left=empty,keep_area=int(sums[keep]))
        self.save(fig,"04_paired_crop","同一张阳性原图，裁剪后可能没有缺陷",["severstal/data.py:SteelDataset.__getitem__",f"raw/data/severstal/train_images/{stem}.jpg"],
                  "裁剪会改变实际看到的标签；几何变换必须同时应用到图像和掩膜。", "本次用验证样本重画的教学示意，窗口按有无缺陷选择；不是历史训练批次，也未将该样本用于训练。", "教学示意 · 非历史训练画面 · 2026-10-24")
        toy=np.zeros((4,6),dtype=np.uint8);toy[1:,1]=1;toy[1:4,3]=1;toy[0,4]=1
        flat=toy.reshape(-1,order="F"); padded=np.r_[0,flat,0]; edges=np.flatnonzero(padded[1:]!=padded[:-1]); starts=edges[::2]+1;lengths=edges[1::2]-edges[::2]
        rle=" ".join(str(x) for pair in zip(starts,lengths) for x in pair)
        assert np.array_equal(decode(rle,4,6),toy)
        fig,axs=plt.subplots(1,2,figsize=(14,7));fig.subplots_adjust(top=.84,bottom=.16,wspace=.32)
        for ax,matrix,title in zip(axs,[np.arange(1,25).reshape(4,6,order="F"),toy],["从 1 开始，沿每列向下编号","有缺陷为 1，其余为 0"]):
            ax.imshow(toy,cmap="Blues",vmin=0,vmax=1)
            for y in range(4):
                for x in range(6): ax.text(x,y,str(matrix[y,x]),ha="center",va="center",color="white" if toy[y,x] else "black",fontsize=16)
            ax.set_xticks(range(6),range(1,7));ax.set_yticks(range(4),range(1,5));ax.set_xlabel("列");ax.set_ylabel("行");ax.set_title(title)
        fig.text(.5,.08,f"RLE = {rle}       每两个数表示：起点、连续长度",ha="center",fontsize=17)
        self.save(fig,"05_rle","RLE 怎样恢复像素标签",["severstal/data.py:decode_rle"],"先减去 1 得到数组起点，再按列恢复二维掩膜；展开顺序错误会使位置错乱。","4×6 人工小网格，用于解释编码；并非真实钢板样本。","人工网格示意 · 非真实样本 · 2026-10-24")

    def training(self):
        fig,axs=plt.subplots(2,2,figsize=(16,10));fig.subplots_adjust(top=.89,bottom=.12,hspace=.34,wspace=.24)
        extract=[lambda r:r['metrics']['nonempty_dice'],lambda r:r['metrics']['classes']['2']['nonempty_dice'],lambda r:r['metrics']['classes']['2']['false_negative_pairs'],lambda r:r['metrics']['classes']['2']['false_positive_pairs']]
        titles=["四类非空 Dice：正式选模依据","类别 2 阳性 Dice：轮廓质量","类别 2 空预测漏检数 / 49","类别 2 阴性误报数"]
        for mi,(member,rows) in enumerate(self.histories.items()):
            epochs=[r['epoch'] for r in rows]; selected=self.summary['members'][member]['epoch']
            assert epochs==list(range(1,len(rows)+1))
            for ax,fn,title in zip(axs.flat,extract,titles):
                values=[fn(r) for r in rows];ax.plot(epochs,values,color=COLORS[mi],label=f"{member} · 正式第{selected}轮",lw=2)
                ax.scatter([selected],[values[selected-1]],color=COLORS[mi],s=90,marker="*",zorder=5)
                ax.set_title(title);ax.set_xlabel("epoch");ax.grid(alpha=.2)
        axs[0,0].legend(fontsize=11);axs[1,0].set_ylim(-1,52)
        self.save(fig,"06_training","完整训练曲线：学习进展与选模时点",[f"raw/runs/{r}/history.jsonl" for r in RUNS.values()],
                  "A/B/C 分别训练 30/18/30 轮，正式选中 29/11/29。类别 2 的学习在早期明显滞后。", "原始验证输出，尚未施加共同面积过滤；空预测漏检不包括预测非空但位置错误的样本。星号是选中权重。")
        rows=read_rows(self.raw/'runs/E02-r02/history.jsonl');epochs=[r['epoch'] for r in rows]
        fig,axs=plt.subplots(1,2,figsize=(16,7));fig.subplots_adjust(top=.85,bottom=.17,wspace=.24)
        axs[0].plot(epochs,[r['metrics']['macro_f1'] for r in rows],label="四类 macro-F1",color=COLORS[0],lw=2)
        axs[0].plot(epochs,[r['metrics']['class_f1'][1] for r in rows],label="类别 2 F1",color=COLORS[1],lw=2)
        axs[1].plot(epochs,[r['metrics']['tp'][1] for r in rows],label="类别 2 真阳性 / 49",color=COLORS[0],lw=2)
        axs[1].plot(epochs,[r['metrics']['fp'][1] for r in rows],label="类别 2 误报",color=COLORS[1],lw=2)
        for ax in axs:
            ax.axvline(28,color="#444444",ls="--",label="正式第 28 轮");ax.axvline(27,color="#999999",ls=":",label="额外保存第 27 轮")
            ax.set_xlabel("epoch");ax.legend(fontsize=10);ax.grid(alpha=.2)
        axs[0].set_title("按 macro-F1 选模");axs[1].set_title("综合 F1 与稀有类召回并非同一目标")
        self.save(fig,"07_classifier","分类器的选择：综合表现与类别 2 召回",["raw/runs/E02-r02/history.jsonl","../../决策记录.md#d43"],
                  "正式使用第 28 轮；第 27 轮因类别 2 召回较高被额外保留，但没有用于主门控对照。", "展示已有训练记录；不能把分类正确率直接当作分割定位质量。")
        crop=read_json('docs/results/E06-r01/crop_exposure.json');x=np.arange(4)
        fig,axs=plt.subplots(1,2,figsize=(16,7));fig.subplots_adjust(top=.84,bottom=.16,wspace=.25)
        for offset,key,label,color in [(-.2,'positive_images','训练阳性原图',COLORS[2]),(.2,'expected_positive_crops_per_epoch','每轮阳性裁剪期望',COLORS[1])]:
            bars=axs[0].bar(x+offset,[r[key] for r in crop],width=.38,label=label,color=color);axs[0].bar_label(bars,fmt='%.1f',fontsize=10,padding=3)
        axs[0].set_xticks(x,[f"类{i}" for i in range(1,5)]);axs[0].set_ylabel("数量");axs[0].legend();axs[0].set_ylim(0,4900)
        bars=axs[1].bar(x,[100*r['mean_positive_crop_probability_given_positive_image'] for r in crop],color=COLORS)
        axs[1].bar_label(bars,fmt='%.1f%%',padding=4);axs[1].set_xticks(x,[f"类{i}" for i in range(1,5)]);axs[1].set_ylim(0,80);axs[1].set_ylabel("阳性原图裁出阳性窗口的概率 / %")
        self.save(fig,"08_crop_exposure","类别 2：198 张阳性原图，对应约 67.2 次阳性裁剪期望",["../E06-r01/crop_exposure.json"],
                  "每图每轮随机取一个 512 像素窗口时，类别 2 阳性原图保留阳性的平均概率约 33.9%。", "对全部 1,089 个合法横向位置统计的期望，非实际采样日志；尚未做采样策略的因果对照。")

    def ensemble(self):
        fig,axs=plt.subplots(2,2,figsize=(17,11));fig.subplots_adjust(top=.89,bottom=.1,hspace=.4,wspace=.22)
        specs=[("全部图像—类别 Dice",lambda r:r['dice'],'YlGnBu','.3f'),("阳性图像—类别 Dice",lambda r:r['nonempty_dice'],'YlGnBu','.3f'),("类别 2 误报数（越低越少）",lambda r:r['classes']['2']['false_positive_pairs'],'Oranges','.0f'),("类别 2 有交集数 / 49",lambda r:r['classes']['2']['overlap_pairs'],'YlGnBu','.0f')]
        for ax,(title,fn,cmap,fmt) in zip(axs.flat,specs):
            values=np.array([[fn(self.summary['combinations'][c][b]) for b in BRANCHES] for c in COMBOS])
            im=ax.imshow(values,cmap=cmap,aspect='auto');threshold=(values.max()+values.min())/2
            for i in range(7):
                for j in range(6): ax.text(j,i,format(values[i,j],fmt),ha='center',va='center',color='white' if np.dot(im.cmap(im.norm(values[i,j]))[:3],[.2126,.7152,.0722])<.5 else 'black',fontsize=11)
            ax.set_xticks(range(6),BRANCH_LABELS,rotation=20);ax.set_yticks(range(7),COMBOS);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.82)
        self.save(fig,"09_ensemble_heatmaps","七组合 × 六种处理：总分、轮廓、误报与定位同时看",["../E03-r02/summary.json"],
                  "soft 的总体 Dice 高，但阳性 Dice 与类别 2 定位降低；最优组合随评价目标而变化。", "统一开发验证集、统一阈值；六处理定义见 E04/D41。颜色标尺在每个子图内独立。")
        mapping={c:{r['image']:r for r in self.comparison[c]['area']} for c in COMBOS}
        names=sorted(mapping['A'],key=lambda n:(mapping['A'][n]['dice'],n))
        assert len(names)==49 and all(set(mapping[c])==set(names) for c in COMBOS)
        self.audit['class2_heatmap_order']=names
        for limit,slug,title in [(49,'10_class2_all49','类别 2 全部 49 张阳性：每张图的融合变化'),(12,'11_class2_weak12','A 最弱的 12 张类别 2 阳性：其他成员带来了什么')]:
            use=names[:limit];values=np.array([[mapping[c][n]['dice'] for c in COMBOS] for n in use])
            fig,ax=plt.subplots(figsize=(12,18 if limit==49 else 8));fig.subplots_adjust(top=.94 if limit==49 else .88,bottom=.08,left=.22,right=.9)
            im=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
            for i in range(limit):
                for j in range(7): ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9 if limit==49 else 12,color='white' if values[i,j]>.55 else 'black')
            ax.set_xticks(range(7),COMBOS);ax.set_yticks(range(limit),[n[:-4]+(' *' if n[:-4] in FIXED else '') for n in use],fontsize=9 if limit==49 else 11)
            fig.colorbar(im,ax=ax,shrink=.65,label='逐图 Dice');ax.set_xlabel('共同轻面积；按 A 的 Dice 升序，同分按文件名；* 为四张预先固定案例')
            self.save(fig,slug,title,["../E03-r02/class2_positive_comparison.json"],
                      "同一组合并非对每张图都更好；8ed6b8054 仅 B 保有预测，其他组合为零。", "全49图版本没有按结果删样本；12图版本按 A 的低 Dice 选取，阅读时配合全量图。")
        change=self.summary['combinations']['ABC']['area']['vs_A'];x=np.arange(4)
        fig,axs=plt.subplots(1,2,figsize=(16,7));fig.subplots_adjust(top=.85,bottom=.16,wspace=.22)
        for ax,keys,title in [(axs[0],('gained_overlap','lost_overlap'),'正确定位：新增与丢失'),(axs[1],('added_false_positives','removed_false_positives'),'阴性误报：新增与移除')]:
            vals=[[change[str(i+1)][k] for i in range(4)] for k in keys]
            for v,offset,color,label in zip(vals,[-.2,.2],[COLORS[1],COLORS[0]],["新增","丢失" if ax is axs[0] else "移除"]):
                bars=ax.bar(x+offset,v,.38,color=color,label=label);ax.bar_label(bars,padding=4)
            ax.set_xticks(x,[f'类别 {i}' for i in range(1,5)]);ax.set_title(title);ax.legend();ax.margins(y=.18);ax.set_ylabel('图像—类别对数量')
        self.save(fig,'12_gains_losses','ABC 相对 A：净提升背后也有新增误报与定位损失',["../E03-r02/summary.json:combinations.ABC.area.vs_A"],
                  "类别 2 新增 1 张定位、无旧定位丢失；误报移除 328、新增 64，净少 264。类别 1、4 的定位净减少。", "定位只要求与真值有像素交集；此图不衡量交集的覆盖质量。")

    def probability_cases(self):
        selected={r['image']:r for r in read_json('docs/results/E03-r02/selected_cases.json')}
        checks=[]
        for stem in FIXED:
            image,masks=self.image_masks(stem);target=masks[1].astype(bool)
            with np.load(self.raw/f'runs/E03-r02/reference_probabilities/{stem}.npz') as archive:
                probs={m:archive[m][1].copy() for m in 'ABC'}
            probs['ABC']=np.mean(np.stack([probs[m] for m in 'ABC']),axis=0)
            union=target.copy()
            for p in probs.values(): union|=p>.5
            xs=np.where(union)[1];left=max(0,int(xs.min())-45);right=min(1600,int(xs.max())+46)
            fig,axs=plt.subplots(1,6,figsize=(17,8)) if right-left<350 else plt.subplots(2,3,figsize=(17,8))
            axs=np.asarray(axs).reshape(2,3);fig.subplots_adjust(top=.85,bottom=.17,hspace=.4,wspace=.22)
            axs[0,0].imshow(image[:,left:right]);axs[0,0].contour(target[:,left:right],levels=[.5],colors=['#6aff68'],linewidths=1);axs[0,0].set_title('原图 · 绿线为真值边界')
            for ax,member in zip([axs[0,1],axs[0,2],axs[1,0],axs[1,1]],['A','B','C','ABC']):
                p=probs[member];pred=p>.5
                if pred.sum()<50: pred[:]=False
                area=int(pred.sum());dice=2*int((pred&target).sum())/(area+int(target.sum()))
                saved=selected[stem+'.jpg']['combinations'][member]
                checks.append(dict(image=stem,member=member,pred_area=area,saved_area=saved['pred_area'],dice=dice,saved_dice=saved['dice']))
                assert area==saved['pred_area'] and abs(dice-saved['dice'])<2e-6,(stem,member,area,saved)
                im=ax.imshow(p[:,left:right],vmin=0,vmax=1,cmap='magma')
                ax.contour(target[:,left:right],levels=[.5],colors=['#6aff68'],linewidths=.8)
                if p.min()<.5<p.max(): ax.contour(p[:,left:right],levels=[.5],colors=['white'],linewidths=.8)
                ax.set_title(f'{member} · Dice {dice:.3f} · 面积 {area}',fontsize=12)
            pred=probs['ABC']>.5
            if pred.sum()<50:pred[:]=False
            mask_image=np.zeros((*target.shape,3),dtype=float);mask_image[target]=[.2,.9,.25];mask_image[pred]=[.9,.2,.7];mask_image[target&pred]=[1,1,1]
            axs[1,2].imshow(mask_image[:,left:right]);axs[1,2].set_title('ABC 二值结果：绿=仅真值\n粉=仅预测，白=交集',fontsize=12)
            for ax in axs.flat:ax.axis('off')
            cax=fig.add_axes([.35,.08,.3,.025]);fig.colorbar(im,cax=cax,orientation='horizontal',label='类别 2 sigmoid 概率 · 白线为 0.5')
            observation='仅 B 有正确交集；概率平均后的弱证据没有越过 0.5，面积过滤前即为空。' if stem=='8ed6b8054' else '不同成员的预测位置和覆盖不同；以真实掩膜及逐图 Dice 判断改善，不只看热力图亮度。'
            self.save(fig,'13_probability_'+stem,'模型看到的概率：'+stem,[f'raw/runs/E03-r02/reference_probabilities/{stem}.npz','../E03-r02/selected_cases.json','raw/data/severstal/train.csv'],
                      observation,f'类别 2；原图横坐标裁取 [{left}, {right})，显示全部真值与各成员 >0.5 预测区域；这是分割概率，不是 Grad-CAM。面积为整张图轻面积过滤后的数量。')
        self.audit['probability_redraw_checks']=checks

    def interpretation(self):
        metric=read_json('docs/results/E03-r02/metric_context.json');total=metric['image_class_pairs'];positive=metric['positive_pairs'];negative=metric['negative_pairs']
        labels=['全空预测','ABC area','ABC soft'];empty=[negative/total];nonempty=[0]
        for b in ['area','soft']:
            r=metric['ABC'][b];empty.append((negative-r['false_positive_pairs_all_classes'])/total);nonempty.append(positive*r['nonempty_dice']/total)
        fig,ax=plt.subplots(figsize=(14,7));fig.subplots_adjust(top=.85,bottom=.2)
        ax.bar(labels,empty,label='真值为空且预测为空的贡献',color='#71889b');ax.bar(labels,nonempty,bottom=empty,label='真值非空的 Dice 贡献',color=COLORS[0])
        for i,(a,b) in enumerate(zip(empty,nonempty)):
            ax.text(i,a+b+.025,f'总体 {a+b:.4f}',ha='center',fontsize=16)
        ax.set_ylim(0,1.12);ax.set_ylabel('对总体 Dice 的加和贡献');ax.legend(loc='upper left');ax.grid(axis='y',alpha=.2)
        fig.text(.5,.08,'验证集 10,056 个图像—类别对：8,638 个为空，1,418 个为阳性',ha='center',fontsize=14)
        self.save(fig,'14_metric_decomposition','为什么全空预测也能得到 0.8590？',["../E03-r02/metric_context.json"],
                  "空标签占多数。soft 大幅减少误报后总体分上升，但阳性 Dice 由 area 的 0.7103 降至 0.6707。", "仅分解本地验证分；不能用空预测参照推算官方测试分布。")
        matched=read_json('docs/results/E04-r01/matched_area_comparison.json')['classes']['2']
        values=[matched['fixed_area_at_no_more_false_positives'],matched['dynamic'],self.summary['combinations']['ABC']['fixed5']['classes']['2'],self.summary['combinations']['ABC']['dynamic']['classes']['2']]
        fig,axs=plt.subplots(1,2,figsize=(16,7));fig.subplots_adjust(top=.85,bottom=.2,wspace=.2)
        ticks=['A 固定244','A 动态','ABC 固定250','ABC 动态']
        for ax,key,title in [(axs[0],'false_positive_pairs','类别 2 误报'),(axs[1],'overlap_pairs','类别 2 有交集 / 49')]:
            bars=ax.bar(ticks,[v[key] for v in values],color=[COLORS[2],COLORS[0],COLORS[2],COLORS[0]]);ax.bar_label(bars,padding=5);ax.set_title(title);ax.margins(y=.2);ax.tick_params(axis='x',rotation=15)
        self.save(fig,'15_matched_control','分类信息的增益，要和固定面积控制一起看',["../E04-r01/matched_area_comparison.json","../E03-r02/summary.json"],
                  "A 上相同 466 个误报时，动态多保留 2 张弱定位；ABC 上两分支均为 273 个误报、48 张定位，且逐图结果相同。", "A 的固定244为事后匹配误报的诊断阈值；ABC 固定250来自既定5倍面积控制，不是同一阈值搜索实验。")
        scores=read_json('docs/results/E07-r01/kaggle_scores.json');x=np.arange(3)
        fig,ax=plt.subplots(figsize=(14,7));fig.subplots_adjust(top=.84,bottom=.2)
        for offset,branch,color in [(-.2,'area',COLORS[2]),(.2,'soft',COLORS[0])]:
            record=next(r for r in scores['submissions'] if r['branch']==branch)
            values=[self.summary['combinations']['ABC'][branch]['dice'],record['public_score'],record['private_score']]
            bars=ax.bar(x+offset,values,.38,label=branch,color=color);ax.bar_label(bars,fmt='%.5f',padding=5,fontsize=13)
        ax.set_xticks(x,['本地开发验证','官方 Public','官方 Private']);ax.set_ylim(0,1.08);ax.set_ylabel('含空图像—类别平均 Dice');ax.legend();ax.grid(axis='y',alpha=.2)
        self.save(fig,'16_official_scores','冻结分支接受官方测试：本地结果不等于测试成绩',["../E07-r01/kaggle_scores.json","../E03-r02/summary.json"],
                  "soft 的 Private 为 0.87003，高于 area 的 0.56641；两分支都存在本地与官方得分差距。", "三个横轴为不同数据分区；本地参与选模与开发，官方为赛后提交得分，无追认比赛名次或隐藏集逐类指标。")
        fig,axs=plt.subplots(1,2,figsize=(16,7));fig.subplots_adjust(top=.85,bottom=.16,wspace=.25)
        costs={}
        for i,(member,rows) in enumerate(self.histories.items()):
            times=np.array([r['train_seconds']+r['val_seconds'] for r in rows])/60;epochs=[r['epoch'] for r in rows]
            costs[member]=dict(completed_epochs=len(rows),recorded_train_val_minutes=float(times.sum()))
            axs[0].plot(epochs,times,label=member,color=COLORS[i],lw=2)
            axs[1].plot(epochs,times.cumsum(),label=f'{member} · {times.sum():.1f} 分钟',color=COLORS[i],lw=2)
        for ax in axs:ax.set_xlabel('epoch');ax.legend();ax.grid(alpha=.2)
        axs[0].set_title('每完整轮：训练 + 验证');axs[0].set_ylabel('分钟');axs[1].set_title('完整轮次累计耗时');axs[1].set_ylabel('分钟')
        self.audit['recorded_costs']=costs
        self.save(fig,'17_actual_cost','设备上的实际耗时：保留波动，不用单一估计代替',[f'raw/runs/{r}/history.jsonl' for r in RUNS.values()],
                  "各轮耗时明显波动；C 完整30轮累计时间最长。历史记录存在热降频和一次暂停恢复。", "按完整轮次日志加总，排除暂停间隔、准备阶段和被中断的未完成轮次；不同设备状态和 batch，不能作纯架构速度结论。")

    def objective_tradeoff(self):
        record=read_json('docs/results/materials_20261024/abc_soft_class2_losses.json')
        fig=plt.figure(figsize=(17,11));grid=fig.add_gridspec(2,3,height_ratios=[1,1.5],hspace=.45)
        fig.subplots_adjust(top=.88,bottom=.13,left=.15,right=.95)
        area=self.summary['combinations']['ABC']['area'];soft=self.summary['combinations']['ABC']['soft']
        specs=[('总体 Dice（含空标签）',[area['dice'],soft['dice']],'.4f',1.08),
               ('类别 2 阴性误报',[area['classes']['2']['false_positive_pairs'],soft['classes']['2']['false_positive_pairs']],'.0f',440),
               ('类别 2 正确定位 / 49',[area['classes']['2']['overlap_pairs'],soft['classes']['2']['overlap_pairs']],'.0f',56)]
        for i,(title,values,fmt,limit) in enumerate(specs):
            ax=fig.add_subplot(grid[0,i]);bars=ax.bar(['area','soft'],values,color=[COLORS[2],COLORS[0]])
            ax.bar_label(bars,fmt='%'+fmt,padding=5,fontsize=14);ax.set_title(title);ax.set_ylim(0,limit)
        cases=record['cases'];ax=fig.add_subplot(grid[1,:]);y=np.arange(len(cases))
        bars=ax.barh(y,[r['area_dice'] for r in cases],color=COLORS[2],label='area：原有分割')
        ax.bar_label(bars,fmt='%.3f',padding=5,fontsize=12)
        ax.scatter(np.zeros(len(cases)),y,color='#ba4646',s=60,zorder=5,label='soft：全部清空，Dice=0')
        ax.set_yticks(y,[r['image'][:-4]+'   q='+format(r['classifier_probability'],'.4f') for r in cases],fontsize=11)
        ax.invert_yaxis();ax.set_xlim(-.03,1);ax.set_xlabel('逐图 Dice');ax.legend(loc='lower right')
        ax.set_title('全部 6 张新增漏检：分类概率 q ≤ 0.5，p×q 无法通过 >0.5 的像素阈值',fontsize=14)
        self.save(fig,'18_score_vs_misses','总分提高的代价：六张原有正确定位被清空',
                  ['docs/results/materials_20261024/abc_soft_class2_losses.json','docs/results/E03-r02/summary.json'],
                  '本地总体 Dice 0.7554→0.9442，类别2误报374→2；同时正确定位48→42。六张原有分割Dice为0.428至0.815，soft后均为0。',
                  '同一开发验证集上 area/soft 的对照；六张为全部新增定位丢失，不是挑选的案例。q来自正式第28轮分类器。官方隐藏集没有逐类标签，本图不代表官方漏检数。')

    def write_gallery(self):
        self.audit['generated_figures']=len(self.cards)
        (self.output/'figure_manifest.json').write_text(json.dumps({'figures':self.cards,'verification':self.audit},ensure_ascii=False,indent=2)+'\n')
        intro='# 展示图册\n\n2026-10-24 从已保存数据与实验记录重绘；没有新增训练或推理。PNG 用于插图，SVG 用于缩放；原图像素、真值和概率均来自已有材料。\n\nA：U-Net＋ResNet34；B：FPN＋SE-ResNeXt50；C：U-Net＋EfficientNet-B3。`raw/` 指仓库下的 `runs/defense_materials_20261024/raw/`，源码与配置路径以仓库为根。\n\n'
        md=[intro]
        cards=[]
        for c in self.cards:
            slug=c['id'];md.append(f"## {c['title']}\n\n![{c['title']}]({slug}.png)\n\n- 观察：{c['observation']}\n- 口径：{c['scope']}\n- 来源："+'、'.join('`'+s+'`' for s in c['sources'])+f"\n- 文件：[PNG]({slug}.png) · [SVG]({slug}.svg)\n\n")
            cards.append(f'<section id="{slug}"><h2>{html.escape(c["title"])}</h2><a href="{slug}.png"><img loading="lazy" src="{slug}.png" alt="{html.escape(c["title"])}"></a><p>{html.escape(c["observation"])}</p><p class="scope">{html.escape(c["scope"])}</p><details><summary>数据来源与文件</summary><p>{html.escape("；".join(c["sources"]))}</p><a href="{slug}.png">PNG</a> · <a href="{slug}.svg">SVG</a></details></section>')
        md.append('## 已有案例与官方凭证\n\n')
        existing=[('分类器误删正确分割','../../E04-r01/cases/268446e17.png'),('删除阴性误报','../../E04-r01/cases/91222b4c9.png'),('成功分割','../../E04-r01/cases/163ff3547.png'),('官方成绩原截图','../../E07-r01/kaggle_official_scores.png')]
        for title,path in existing:
            assert (self.output/path).is_file(),path
            md.append(f'- [{title}]({path})\n')
            cards.append(f'<section><h2>{title} · 既有原图</h2><a href="{path}"><img loading="lazy" src="{path}" alt="{title}"></a></section>')
        (self.output/'图册.md').write_text(''.join(md))
        toc=''.join(f'<a href="#{c["id"]}">{i+1}. {html.escape(c["title"])}</a>' for i,c in enumerate(self.cards))
        page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Severstal 展示图册</title><style>body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;margin:0;background:#eef2f4;color:#20303e}main{max-width:1200px;margin:auto;padding:30px}h1{font-size:32px}section{background:white;padding:24px;margin:26px 0;border-radius:14px;box-shadow:0 4px 18px #20303e12}img{width:100%;height:auto}p{line-height:1.7}.scope{color:#617080}nav{display:grid;grid-template-columns:1fr 1fr;gap:12px}a{color:#126b7a}details{font-size:14px;overflow-wrap:anywhere}@media(max-width:700px){nav{grid-template-columns:1fr}main{padding:15px}section{padding:10px}}</style><main><h1>Severstal 展示图册</h1><p>2026-10-24 · 已有证据重绘。点击图像打开原尺寸；每图附观察、统计口径和来源。<br>这是展示素材库，尚未确定最终 slides 的选图与顺序。<br>A：U-Net＋ResNet34；B：FPN＋SE-ResNeXt50；C：U-Net＋EfficientNet-B3。<br>来源中的 raw/ 指 runs/defense_materials_20261024/raw/；源码与配置路径以仓库为根。</p><nav>'+toc+'</nav>'+''.join(cards)+'</main></html>'
        (self.output/'index.html').write_text(page)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',default='runs/defense_materials_20261024/raw')
    parser.add_argument('--output',default='docs/results/materials_20261024/figures')
    args=parser.parse_args()
    figures=EvidenceFigures(args.raw,args.output)
    for method in ('dataset','example_details','processing','training','ensemble','probability_cases','interpretation','objective_tradeoff'):
        getattr(figures,method)()
        print(f'{method}: saved {len(figures.cards)} figures',flush=True)
    figures.write_gallery()
    print(f'Complete: {len(figures.cards)} PNG/SVG pairs in {args.output}',flush=True)


if __name__=='__main__':
    main()
