"""Optional integration test for live semantic corrections on a real Spark DSG."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


spark_dsg = pytest.importorskip("spark_dsg")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from daaam.grounding.models import ObjectAnnotation  # noqa: E402
from daaam.scene_graph.services import SceneGraphService  # noqa: E402
from spark_dsg import (  # noqa: E402
    DsgLayers,
    DynamicSceneGraph,
    KhronosObjectAttributes,
    NodeSymbol,
)


def test_live_correction_updates_real_dsg_and_acknowledges_delivery():
    service = SceneGraphService(
        REPOSITORY_ROOT / "config" / "labels_pseudo.yaml",
        REPOSITORY_ROOT / "config" / "labels_pseudo.csv",
        defer_dsg_processing=False,
        enable_background_objects=False,
    )
    graph = DynamicSceneGraph()
    attributes = KhronosObjectAttributes()
    attributes.semantic_label = 42
    attributes.position = [1.0, 2.0, 3.0]
    assert graph.add_node(
        DsgLayers.OBJECTS,
        NodeSymbol("O", 1),
        attributes,
    )
    service.set_scene_graph(graph)

    service.store_correction(
        ObjectAnnotation(
            semantic_id=42,
            semantic_label="chair",
            confidence=0.9,
        )
    )

    node = next(iter(graph.get_layer(DsgLayers.OBJECTS).nodes))
    stats = service.get_correction_stats()
    assert node.attributes.metadata.get()["description"] == "chair"
    assert stats["applied_corrections"] == 1
    assert stats["pending_corrections"] == 0
    assert stats["application_events"] == 1
