import torch
from omegaconf import OmegaConf

from caustics_backscatter.config import load_config
from caustics_backscatter.data.augment import Augmenter, ColorJitter, hsv_to_rgb, rgb_to_hsv
from caustics_backscatter.models import BackscatterNet, SMPDensePredictor, set_bn_momentum
from caustics_backscatter.registry import MODELS


def test_hsv_roundtrip():
    x = torch.rand(4, 3, 8, 8)
    assert torch.allclose(hsv_to_rgb(rgb_to_hsv(x)), x, atol=1e-5)


def test_neutral_jitter_is_identity():
    x = torch.rand(2, 2, 3, 8, 8)
    jitter = ColorJitter()
    assert torch.allclose(jitter(x), x)


def test_pair_jitter_shared_across_views_and_per_image_independent():
    torch.manual_seed(0)
    view = torch.rand(4, 1, 3, 16, 16)
    images = view.repeat(1, 2, 1, 1, 1)
    aug = Augmenter(
        jitter={"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.05},
        per_image_jitter={"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.05},
    )
    paired = aug.pair_jitter(images)
    assert torch.allclose(paired[:, 0], paired[:, 1], atol=1e-6)
    assert not torch.allclose(paired, images)
    single = aug.per_image_jitter(images)
    assert not torch.allclose(single[:, 0], single[:, 1])
    assert paired.min() >= 0 and paired.max() <= 1


def test_models_shapes_and_bn_momentum():
    causticsnet = SMPDensePredictor(encoder_name="resnet18", encoder_weights=None)
    set_bn_momentum(causticsnet, 0.01)
    assert all(m.momentum == 0.01 for m in causticsnet.modules() if isinstance(m, torch.nn.BatchNorm2d))
    out = causticsnet(torch.rand(2, 2, 3, 64, 128))
    assert out.shape == (2, 2, 3, 64, 128)

    backscatternet = BackscatterNet(encoder_name="resnet18", encoder_weights=None, b_inf_init=0.3, beta_init=1.0)
    backscatternet.eval()
    b_inf, beta = backscatternet(torch.rand(2, 3, 3, 64, 128))
    assert b_inf.shape == beta.shape == (2, 3, 3)
    assert torch.allclose(b_inf, torch.full_like(b_inf, 0.3), atol=1e-5)
    assert torch.allclose(beta, torch.ones_like(beta), atol=1e-5)


def test_configs_build_models():
    expected_gaps = {"causticsnet": (2, 5), "backscatternet": (3, 20)}
    for name, (gap_min, gap_max) in expected_gaps.items():
        cfg = load_config(f"configs/{name}.yaml", ["model.encoder_weights=null", "model.encoder_name=resnet18"])
        assert (cfg.views.gap_min, cfg.views.gap_max) == (gap_min, gap_max)
        model = MODELS.build(cfg.model)
        assert isinstance(model, torch.nn.Module)
        assert OmegaConf.select(cfg, "geometry.max_hop") == 4
