"""Published ORION marker order and HE normalization constants."""
IMAGENET_STATS = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

CHANNELS = [
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO",
    "CD20", "PDL1", "CD3e", "CD163", "ECadherin", "Ki67", "Pan-CK", "SMA",
]

MIF_FULL_CHANNELS = [
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO",
    "CD20", "PDL1", "CD3e", "CD163", "ECadherin", "PD-1", "Ki67", "Pan-CK", "SMA",
]

MIF_SELECT = [i for i in range(len(MIF_FULL_CHANNELS)) if i != 13]
