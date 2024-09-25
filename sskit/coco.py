from xtcocotools.cocoeval import COCOeval
import numpy as np
from sskit import image_to_ground

class LocSimCOCOeval(COCOeval):
    def get_img_pos(self, dt):
        return [np.array(det['keypoints']).reshape(-1,3)[1, :2] for det in dt]

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 or len(dt) == 0:
            return []
        inds = np.argsort([-d[self.score_key] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        img = self.cocoGt.loadImgs(int(imgId))[0]
        img_pos_dt = np.array(self.get_img_pos(dt))
        w, h = np.float32(img['width']), np.float32(img['height'])
        nimg_pos_dt = ((img_pos_dt - (w/2, h/2)) / w).astype(np.float32)
        bev_dt = image_to_ground(img['camera_matrix'], img['undist_poly'], nimg_pos_dt)[:, :2]
        bev_gt = np.array([det['position_on_pitch'] for det in gt])

        aa, bb = np.meshgrid(bev_gt[:,0], bev_dt[:,0])
        dist2 = (aa - bb) ** 2
        aa, bb = np.meshgrid(bev_gt[:,1], bev_dt[:,1])
        dist2 += (aa - bb) ** 2

        tau = 1
        locsim = np.exp(np.log(0.05) * dist2 / tau**2)
        return locsim

    def accumulate(self, p=None):
        if p is None:
            p = self.params
        super().accumulate(p)

        iou = p.iouThrs == 0.5
        area = p.areaRngLbl.index('all')
        dets = np.argmax(p.maxDets)

        precision = np.squeeze(self.eval['precision'][iou, :, 0, area, dets])
        scores = np.squeeze(self.eval['scores'][iou, :, 0, area, dets])
        recall = p.recThrs
        f1 = 2 * precision * recall / (precision + recall)

        self.eval['precision_50'] = precision
        self.eval['recall_50'] = recall
        self.eval['f1_50'] = f1
        self.eval['scores_50'] = scores

    def summarize(self):
        super().summarize()
        if hasattr(self.params, 'score_threshold'):
            threshold = self.params.score_threshold
        else:
            i = self.eval['f1_50'].argmax()
            threshold = (self.eval['scores_50'][i] + self.eval['scores_50'][i+1]) / 2
        i = np.searchsorted(-self.eval['scores_50'], -threshold, 'right') - 1
        stats = [self.eval['precision_50'][i], self.eval['recall_50'][i], self.eval['f1_50'][i], threshold]
        self.stats = np.concatenate([self.stats, stats])


class BBoxLocSimCOCOeval(LocSimCOCOeval):
    def get_img_pos(self, dt):
        def bbox_ground(x, y, w, h):
            return (x + w/2, y + h)
        return [bbox_ground(*det['bbox']) for det in dt]

