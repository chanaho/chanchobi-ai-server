from fastapi import APIRouter

router = APIRouter()

@router.get("/map")
def map_data(lat: float, lon: float):

    return {
        "success": True,
        "center": {
            "lat": lat,
            "lon": lon
        },
        "zoom": 14
    }