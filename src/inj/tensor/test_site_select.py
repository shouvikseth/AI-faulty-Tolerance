
# src/inj/tensor/test_site_select.py
import torch
from src.inj.tensor.site_select import SiteSelector, SitePlan

def main():
    sel1 = SiteSelector(seed=123, p_event=0.25, bit_width=32)
    sel2 = SiteSelector(seed=123, p_event=0.25, bit_width=32)

    shape = torch.Size([64, 64])
    p1 = sel1.plan(shape, layer_id="Linear[-1]", fwd_idx=1)
    p2 = sel2.plan(shape, layer_id="Linear[-1]", fwd_idx=1)

    print("same plan:", p1 == p2)
    print("plan:", p1)

    sel3 = SiteSelector(seed=124, p_event=0.25, bit_width=32)
    p3 = sel3.plan(shape, "Linear[-1]", 1)
    print("different seed → different plan:", p3 != p1)

if __name__ == "__main__":
    main()
