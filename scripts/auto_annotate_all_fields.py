import sys
import json
import re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(".").resolve()))

from scripts.group_visual_bills import groups

AUTO_HW_DIR = Path("annotations/reviewed/auto_handwritten")
DIGITS_OUT_DIR = Path("annotations/reviewed/digits")
HW_OUT_DIR = Path("annotations/reviewed/handwritten")

DIGITS_OUT_DIR.mkdir(parents=True, exist_ok=True)
HW_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ground truth dictionary per group/bill
# Each group has ground truth dictionary for its fields
GROUP_GROUND_TRUTH = {
    # Group 1: Slip 85 (Pink)
    1: {
        "BOOK_NO": "3232",
        "SLIP_NO": "85",
        "DATE": "29/10/25",
        "COMMISSION_AGENT": "ముత్యాల ఉపేందర్",
        "FARMER_NAME": "M.వినీల్",
        "VILLAGE": "లచ్చగూడెం",
        "MANDAL": "ఇల్లెందు",
        "COMMODITY": "పత్తి",
        "BUYER_NAME": "శ్రీ వడ్డాల కృష్ణ కాటన్ ట్రేడ్స్",
        "NO_OF_BAGS": "23",
        "RATE_PER_QTL": "5800",
        "భర్తీ/వసూలు కిలోల కీ.గ్రా.(Weight in kg)": "1000",
        "COMMISSION_ADTI": "-",
        "HAMALI": "414",
        "ఆసామికి చెల్లించవలసినవి": "1585",
        "మొత్తం": "58575",
        "మొత్తము": "56990",
        "క్రమ సంఖ్య(Serial Number)": "1",
    },
    # Group 2: Slip 96 (Pink)
    2: {
        "BOOK_NO": "103169",
        "SLIP_NO": "96",
        "DATE": "29/10/2025",
        "COMMISSION_AGENT": "కూచిపూడి నరసింహారావు",
        "FARMER_NAME": "T.నరసింహారావు",
        "VILLAGE": "కడముట్ట",
        "COMMODITY": "మిర్చి",
        "BUYER_NAME": "అంకుర్ ట్రేడర్స్",
        "NO_OF_BAGS": "23",
        "RATE_PER_QTL": "13300",
        "COMMISSION_ADTI": "112917",
        "HAMALI": "2672",
        "మొత్తము": "110820",
        "ఆసామికి చెల్లించవలసినవి Amount payable to the owner/seller": "2672",
    },
    # Group 3: Slip 82 (Yellow)
    3: {
        "BOOK_NO": "2939",
        "SLIP_NO": "82",
        "DATE": "3.9.25",
        "COMMISSION_AGENT": "తారకేంద్ర కుమార్",
        "FARMER_NAME": "B.అంజి రెడ్డి",
        "VILLAGE": "మేడిపల్లి",
        "COMMODITY": "మిర్చి",
        "BUYER_NAME": "monica spices",
        "NO_OF_BAGS": "34",
        "RATE_PER_QTL": "13000",
        "COMMISSION_ADTI": "1793",
        "HAMALI": "574",
        "మొత్తము": "179380.00",
        "ఆసామికి చెల్లించవలసినవి Amount payable to the owner/seller": "9267",
    },
    # Group 4: Slip 81 (Yellow)
    4: {
        "BOOK_NO": "2939",
        "SLIP_NO": "81",
        "DATE": "3.9.25",
        "COMMISSION_AGENT": "తారకేంద్ర కుమార్",
        "FARMER_NAME": "V.వెంకటేశ్వర్లు",
        "VILLAGE": "మ.దుర్గం",
        "COMMODITY": "మిర్చి",
        "BUYER_NAME": "chs Na Va Padmini Sales Pvt Ltd",
        "NO_OF_BAGS": "20",
        "RATE_PER_QTL": "13500",
        "COMMISSION_ADTI": "1062",
        "HAMALI": "380",
        "మొత్తము": "106145.00",
        "ఆసామికి చెల్లించవలసినవి Amount payable to the owner/seller": "1389",
    },
    # Group 5: Slip 78 (Yellow)
    5: {
        "BOOK_NO": "2939",
        "SLIP_NO": "78",
        "DATE": "3.9.25",
        "COMMISSION_AGENT": "తారకేంద్ర కుమార్",
        "FARMER_NAME": "V.వెంకటేశ్వర్లు",
        "VILLAGE": "కెనగండి",
        "COMMODITY": "మిర్చి",
        "BUYER_NAME": "chs Shashta Traders",
        "NO_OF_BAGS": "58",
        "RATE_PER_QTL": "7700",
        "COMMISSION_ADTI": "1670",
        "HAMALI": "928",
        "మొత్తము": "167013.00",
        "ఆసామికి చెల్లించవలసినవి Amount payable to the owner/seller": "2598",
    },
    # Group 6: Slip 12 (Pink)
    6: {
        "BOOK_NO": "3438",
        "SLIP_NO": "12",
        "DATE": "6.10.25",
        "COMMISSION_AGENT": "తారకేంద్ర కుమార్",
        "FARMER_NAME": "B.సైదులు",
        "VILLAGE": "మహల్ గూడ",
        "COMMODITY": "మిర్చి",
        "BUYER_NAME": "chs Shaesta Trader",
        "NO_OF_BAGS": "135",
        "RATE_PER_QTL": "7500",
        "COMMISSION_ADTI": "4168",
        "HAMALI": "2160",
        "మొత్తము": "416850.00",
        "ఆసామికి చెల్లించవలసినవి Amount payable to the owner/seller": "6328",
    },
    # Group 7: Slip 56 (Pink)
    7: {
        "BOOK_NO": "2921",
        "SLIP_NO": "56",
        "DATE": "16/10/25",
        "COMMISSION_AGENT": "వెంకట నాగరాజు",
        "FARMER_NAME": "K.రాంబాబు",
        "VILLAGE": "మర్రిపాడు",
        "COMMODITY": "మిర్చి",
        "BUYER_NAME": "SS Agri Traders",
        "NO_OF_BAGS": "25",
        "RATE_PER_QTL": "6500",
        "COMMISSION_ADTI": "1406",
        "HAMALI": "575",
        "మొత్తము": "68955.00",
        "ఆసామికి చెల్లించవలసినవి Amount payable to the owner/seller": "1981.00",
    },
    # Group 8: Slip 23 (Pink)
    8: {
        "BOOK_NO": "3011",
        "SLIP_NO": "23",
        "DATE": "20.11.25",
        "COMMISSION_AGENT": "కూచిపూడి నరసింహారావు",
        "FARMER_NAME": "T.మధుసూదన్ రెడ్డి",
        "VILLAGE": "కలకోట",
        "COMMODITY": "మిర్చి",
        "BUYER_NAME": "శ్రీను ట్రేడర్స్",
        "NO_OF_BAGS": "87",
        "RATE_PER_QTL": "5300",
        "COMMISSION_ADTI": "4178",
        "HAMALI": "1566",
        "మొత్తము": "208926.00",
        "ఆసామికి చెల్లించవలసినవి Amount payable to the owner/seller": "5744.0",
    },
    # Group 9: Slip 75 (Pink)
    9: {
        "BOOK_NO": "3468",
        "SLIP_NO": "75",
        "DATE": "22/10/25",
        "COMMISSION_AGENT": "తారకేంద్ర కుమార్",
        "FARMER_NAME": "P.రవి",
        "VILLAGE": "ఎర్రుపాలెం",
        "COMMODITY": "మిర్చి",
        "BUYER_NAME": "శ్రీ శ్రీనివాస ట్రేడర్స్",
        "NO_OF_BAGS": "112",
        "RATE_PER_QTL": "1390",
        "COMMISSION_ADTI": "13261",
        "HAMALI": "2240",
        "మొత్తము": "650586",
        "ఆసామికి చెల్లించవలసినవి Amount payable to the owner/seller": "15505",
    },
    # Group 10: Slip 21 (Pink)
    10: {
        "BOOK_NO": "3468",
        "SLIP_NO": "21",
        "DATE": "27/10/25",
        "COMMISSION_AGENT": "కూచిపూడి నరసింహారావు",
        "FARMER_NAME": "V.వెంకటేశ్వర్లు",
        "VILLAGE": "గంగాపురం",
        "COMMODITY": "మిర్చి",
        "BUYER_NAME": "B.కుమార్ ట్రేడర్స్",
        "NO_OF_BAGS": "37",
        "RATE_PER_QTL": "8000",
        "COMMISSION_ADTI": "1863",
        "HAMALI": "740",
        "మొత్తము": "124240.00",
        "ఆసామికి చెల్లించవలసినవి Amount payable to the owner/seller": "2603.00",
    }
}

# Map each source image to its group ID
file_to_group = {}
for g_idx, g_files in enumerate(groups, 1):
    for fn in g_files:
        file_to_group[fn] = g_idx

# Process all auto crops
auto_records = [json.loads(p.read_text(encoding="utf-8")) for p in AUTO_HW_DIR.glob("*.json")]
print(f"Assigning verified labels to {len(auto_records)} auto crops across 10 groups...")

digits_saved = 0
hw_saved = 0

for r in auto_records:
    src_fn = r["source_image"]
    g_id = file_to_group.get(src_fn)
    if not g_id or g_id not in GROUP_GROUND_TRUTH:
        continue
    
    gt_map = GROUP_GROUND_TRUTH[g_id]
    f_name = r["field_name"]
    text = gt_map.get(f_name, "")
    
    r["text"] = text
    r["group_id"] = g_id
    r["status"] = "VERIFIED_AUTO"
    
    if r["type"] == "digits":
        out_p = DIGITS_OUT_DIR / f"{r['id']}.json"
        out_p.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        digits_saved += 1
    else:
        out_p = HW_OUT_DIR / f"{r['id']}.json"
        out_p.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        hw_saved += 1

print(f"Saved {digits_saved} verified digit records to {DIGITS_OUT_DIR}")
print(f"Saved {hw_saved} verified handwritten records to {HW_OUT_DIR}")
