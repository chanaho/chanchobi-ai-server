import os
import requests
import xml.etree.ElementTree as ET


PSIS_URL = "http://psis.rda.go.kr/openApi/service.do"


def _local_name(tag):
    """
    XML namespace가 있어도 태그 이름만 추출
    """
    return tag.split("}", 1)[-1]


def _get_child_text(element, name):
    """
    XML item 안에서 지정된 필드의 값을 가져옴
    """
    for child in list(element):
        if _local_name(child.tag) == name:
            return (child.text or "").strip()

    return ""


def search_psis_pesticides(
    crop_name,
    disease_pest_name
):

    api_key = os.getenv("PSIS_API_KEY")

    if not api_key:

        print(
            "PSIS API KEY NOT FOUND"
        )

        return {
            "status": "NO_API_KEY",
            "items": []
        }


    params = {

        "apiKey": api_key,

        "serviceCode": "SVC01",

        # AA001 = XML
        "serviceType": "AA001",

        "displayCount": 50,

        "startPoint": 1,

        # 작물명 정확히 일치
        "cropName": crop_name,

        "cropCheck": "Y",

        # 적용 병해충
        "diseaseWeedName": disease_pest_name,

        # 유사 병해충 검색하지 않음
        "similarFlag": "N"
    }


    try:

        response = requests.get(
            PSIS_URL,
            params=params,
            timeout=15
        )

        print(
            "PSIS HTTP STATUS:",
            response.status_code
        )

        response.raise_for_status()


        # =========================
        # XML 파싱
        # =========================

        root = ET.fromstring(
            response.content
        )


        items = []


        # XML 내부의 모든 item 검색
        for element in root.iter():

            if _local_name(element.tag) != "item":
                continue


            row = {

                "cropName":
                    _get_child_text(
                        element,
                        "cropName"
                    ),

                "diseaseWeedName":
                    _get_child_text(
                        element,
                        "diseaseWeedName"
                    ),

                "useName":
                    _get_child_text(
                        element,
                        "useName"
                    ),

                "pestiKorName":
                    _get_child_text(
                        element,
                        "pestiKorName"
                    ),

                "pestiBrandName":
                    _get_child_text(
                        element,
                        "pestiBrandName"
                    ),

                "compName":
                    _get_child_text(
                        element,
                        "compName"
                    ),

                "engName":
                    _get_child_text(
                        element,
                        "engName"
                    ),

                "cmpaItmNm":
                    _get_child_text(
                        element,
                        "cmpaItmNm"
                    ),

                "indictSymbl":
                    _get_child_text(
                        element,
                        "indictSymbl"
                    ),

                "applyFirstRegDate":
                    _get_child_text(
                        element,
                        "applyFirstRegDate"
                    ),

                "pestiUse":
                    _get_child_text(
                        element,
                        "pestiUse"
                    ),

                "dilutUnit":
                    _get_child_text(
                        element,
                        "dilutUnit"
                    ),

                "useSuittime":
                    _get_child_text(
                        element,
                        "useSuittime"
                    ),

                "useNum":
                    _get_child_text(
                        element,
                        "useNum"
                    ),

                "pestiCode":
                    _get_child_text(
                        element,
                        "pestiCode"
                    ),

                "diseaseUseSeq":
                    _get_child_text(
                        element,
                        "diseaseUseSeq"
                    ),

                "cropCd":
                    _get_child_text(
                        element,
                        "cropCd"
                    ),

                "cropLrclCd":
                    _get_child_text(
                        element,
                        "cropLrclCd"
                    ),

                "cropLrclNm":
                    _get_child_text(
                        element,
                        "cropLrclNm"
                    )
            }


            # 실제 데이터가 있는 item만 추가
            if any(row.values()):

                items.append(
                    row
                )


        # =========================
        # 결과 확인
        # =========================

        if items:

            print(
                "PSIS XML RESULT:",
                len(items)
            )

            return {
                "status": "OK",
                "items": items
            }


        # =========================
        # 결과 없음 / 오류 메시지 확인
        # =========================

        result_code = ""
        result_msg = ""


        for element in root.iter():

            tag = _local_name(
                element.tag
            )


            if tag == "resultCode":

                result_code = (
                    element.text or ""
                ).strip()


            elif tag == "resultMsg":

                result_msg = (
                    element.text or ""
                ).strip()


        print(
            "PSIS RESULT CODE:",
            result_code
        )

        print(
            "PSIS RESULT MSG:",
            result_msg
        )


        if result_code:

            return {

                "status": "PSIS_ERROR",

                "items": [],

                "resultCode":
                    result_code,

                "resultMsg":
                    result_msg
            }


        return {

            "status": "NO_RESULT",

            "items": []
        }


    except ET.ParseError as e:

        print(
            "PSIS XML PARSE ERROR:",
            e
        )

        print(
            "PSIS RESPONSE START:",
            response.text[:500]
        )

        return {

            "status": "XML_PARSE_ERROR",

            "items": []
        }


    except requests.RequestException as e:

        print(
            "PSIS REQUEST ERROR:",
            e
        )

        return {

            "status": "REQUEST_ERROR",

            "items": []
        }


    except Exception as e:

        print(
            "PSIS UNKNOWN ERROR:",
            e
        )

        return {

            "status": "ERROR",

            "items": []
        }
# =========================
# PSIS 농약 상세정보 조회
# =========================
def get_psis_pesticide_detail(pesti_code, disease_use_seq):
    api_key = os.getenv("PSIS_API_KEY")

    if not api_key:
        print("PSIS API KEY NOT FOUND")
        return {
            "status": "NO_API_KEY",
            "item": None
        }

    params = {
        "apiKey": api_key,
        "serviceCode": "SVC02",
        "serviceType": "AA001",
        "pestiCode": pesti_code,
        "diseaseUseSeq": disease_use_seq
    }

    try:
        response = requests.get(
            PSIS_URL,
            params=params,
            timeout=15
        )

        print(
            "PSIS DETAIL HTTP STATUS:",
            response.status_code
        )

        response.raise_for_status()

        root = ET.fromstring(response.content)

        # SVC02는 <item>이 아니라
        # <service> 바로 아래에 상세 필드가 반환됨
        item = {
            "pestiKorName": _get_child_text(
                root,
                "pestiKorName"
            ),
            "useName": _get_child_text(
                root,
                "useName"
            ),
            "compName": _get_child_text(
                root,
                "compName"
            ),
            "pestiBrandName": _get_child_text(
                root,
                "pestiBrandName"
            ),
            "pestiEngName": _get_child_text(
                root,
                "pestiEngName"
            ),
            "regCpntQnty": _get_child_text(
                root,
                "regCpntQnty"
            ),
            "toxicGubun": _get_child_text(
                root,
                "toxicGubun"
            ),
            "toxicName": _get_child_text(
                root,
                "toxicName"
            ),
            "fishToxicGubun": _get_child_text(
                root,
                "fishToxicGubun"
            ),
            "cropName": _get_child_text(
                root,
                "cropName"
            ),
            "diseaseWeedName": _get_child_text(
                root,
                "diseaseWeedName"
            ),
            "pestiUse": _get_child_text(
                root,
                "pestiUse"
            ),
            "dilutUnit": _get_child_text(
                root,
                "dilutUnit"
            ),
            "useSuittime": _get_child_text(
                root,
                "useSuittime"
            ),
            "useNum": _get_child_text(
                root,
                "useNum"
            )
        }

        # 실제 데이터가 하나라도 있으면 정상
        if any(item.values()):
            print("PSIS DETAIL RESULT: OK")

            return {
                "status": "OK",
                "item": item
            }

        print("PSIS DETAIL RESULT: NO_RESULT")

        return {
            "status": "NO_RESULT",
            "item": None
        }

    except ET.ParseError as e:
        print(
            "PSIS DETAIL XML PARSE ERROR:",
            e
        )

        return {
            "status": "XML_PARSE_ERROR",
            "item": None
        }

    except requests.RequestException as e:
        print(
            "PSIS DETAIL REQUEST ERROR:",
            e
        )

        return {
            "status": "REQUEST_ERROR",
            "item": None
        }

    except Exception as e:
        print(
            "PSIS DETAIL UNKNOWN ERROR:",
            e
        )

        return {
            "status": "ERROR",
            "item": None
        }
# ==============================
# PSIS LOCAL OFFICIAL DB
# ==============================

import json
from pathlib import Path


PSIS_LOCAL_DB_DIR = (
    Path(__file__).resolve().parent.parent / "psis_db"
)


def load_local_psis_pesticides(
    crop_name,
    disease_pest_name
):
    """
    PSIS에서 미리 확보한 공식 등록정보를
    로컬 JSON DB에서 조회한다.

    실시간 PSIS 접속이 필요하지 않다.
    """

    if not crop_name or not disease_pest_name:
        return {
            "status": "INVALID_PARAMETER",
            "items": []
        }

    crop = str(crop_name).strip()
    target = str(disease_pest_name).strip()

    if not PSIS_LOCAL_DB_DIR.exists():
        print(
            "PSIS LOCAL DB DIR NOT FOUND:",
            PSIS_LOCAL_DB_DIR
        )

        return {
            "status": "DB_NOT_FOUND",
            "items": []
        }

    items = []

    for db_file in PSIS_LOCAL_DB_DIR.glob("*.json"):

        try:

            with open(
                db_file,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            if not isinstance(data, list):
                continue

            for item in data:

                item_crop = str(
                    item.get("cropName", "")
                ).strip()

                item_target = str(
                    item.get("diseaseWeedName", "")
                ).strip()

                if (
                    item_crop == crop
                    and item_target == target
                ):
                    items.append(item)

        except Exception as e:

            print(
                "PSIS LOCAL DB READ ERROR:",
                db_file,
                e
            )

    if items:

        print(
            "PSIS LOCAL DB RESULT:",
            crop,
            target,
            len(items)
        )

        return {
            "status": "OK",
            "items": items
        }

    print(
        "PSIS LOCAL DB NO RESULT:",
        crop,
        target
    )

    return {
        "status": "NO_RESULT",
        "items": []
    }
