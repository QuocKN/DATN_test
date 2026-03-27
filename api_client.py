import argparse
import requests
import json
import sys


def call_api(image_path, mode="both", server_url="http://localhost:5000"):
    """Call the prediction API."""
    try:
        # Prepare request
        params = {
            "image": image_path,
            "mode": mode
        }
        
        # Send request
        response = requests.get(f"{server_url}/predict", params=params)
        
        if response.status_code == 200:
            result = response.json()
            print_result(result)
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return None
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {server_url}")
        print("Make sure the API server is running:")
        print("  python api_server.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        return None


def print_result(result):
    """Pretty print API result."""
    if result.get("status") != "success":
        print(f"Error: {result.get('error')}")
        return
    
    print("\n" + "="*50)
    print(f"Inference time: {result['inference_time_seconds']}s on {result['device']}")
    print("="*50)
    
    results = result.get("results", {})
    
    for model_name in ["svm", "knn"]:
        if model_name in results:
            pred = results[model_name]
            print(f"\n=== {model_name.upper()} ===")
            print(f"Ket qua: {'CO drone' if pred['prediction'] == 'drone' else 'KHONG co drone'}")
            print(f"Do tin cay: {pred['confidence']:.4f} ({pred['confidence_percent']})")
    
    print("\n" + "="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Client for drone/noise prediction API"
    )
    parser.add_argument("image", help="Path to spectrogram image")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["svm", "knn", "both"],
        help="Choose model: svm, knn, or both",
    )
    parser.add_argument(
        "--server",
        default="http://localhost:5000",
        help="API server URL",
    )
    
    args = parser.parse_args()
    
    call_api(args.image, mode=args.mode, server_url=args.server)


if __name__ == "__main__":
    main()
