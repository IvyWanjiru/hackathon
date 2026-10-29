"""Verify model loading and HTTP inference locally without starting a server."""
import argparse

from .service import create_app


def smoke(reason_config, matching_index=None):
    client = create_app(reason_config, matching_index).test_client()
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json={"text": ""}).status_code == 400
    assert client.post("/predict", json=[]).status_code == 400
    response = client.post("/predict", json={"text": "The requested running shoes size 42 are out of stock"})
    assert response.status_code == 200, response.get_json()
    result = response.get_json()
    assert result["review_required"] is True
    if result["reason"]:
        assert 0 <= result["reason"]["confidence"] <= 1
    if matching_index:
        assert result["matches"] and result["matches"][0]["requires_review"] is True
    print("PASS: health, input validation, artifact loading, reason inference and optional matching. No application database opened.")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason-config", default="ml_workflow/work/baseline/serving.json")
    parser.add_argument("--matching-index")
    args = parser.parse_args()
    smoke(args.reason_config, args.matching_index)
