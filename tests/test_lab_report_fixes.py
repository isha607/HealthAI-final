import urllib.request
import json

def test_lab_report_table():
    resp = urllib.request.urlopen("http://localhost:8000/api/documents/patient/PAT-1001/report-table")
    data = json.loads(resp.read().decode())
    assert "rows" in data and len(data["rows"]) > 0, "No rows in report-table"
    for r in data["rows"]:
        name = r.get("parameter_name") or r.get("test_name")
        assert name and name != "undefined", f"Invalid parameter name: {name}"
        ref = r.get("reference_interval") or r.get("reference_range")
        assert ref and ref != "undefined", f"Invalid reference range: {ref}"
    print(f"PASS: Report table validated ({len(data['rows'])} parameters, zero undefined)")

def test_lab_report_trends():
    resp2 = urllib.request.urlopen("http://localhost:8000/api/documents/patient/PAT-1001/report-trends")
    data2 = json.loads(resp2.read().decode())
    assert "available_parameters" in data2 and len(data2["available_parameters"]) > 0
    for p in data2["available_parameters"]:
        pname = p.get("name") or p.get("parameter_name") or p.get("test_name")
        assert pname and pname != "undefined", f"Invalid trend param name: {pname}"
    assert "datasets" in data2 and "series" in data2
    for k, ds in data2["datasets"].items():
        assert "data" in ds and "values" in ds
    print(f"PASS: Report trends validated ({len(data2['available_parameters'])} available params, zero undefined)")

def test_lab_report_documents_and_urls():
    resp3 = urllib.request.urlopen("http://localhost:8000/api/documents/patient/PAT-1001/reports")
    data3 = json.loads(resp3.read().decode())
    doc_count = 0
    for rep in data3:
        if rep["has_document"]:
            doc_count += 1
            url = rep["file_url"]
            assert url, f"Missing file_url for report {rep['id']}"
            f_resp = urllib.request.urlopen(f"http://localhost:8000{url}")
            assert f_resp.status == 200, f"Expected 200 for {url}, got {f_resp.status}"
            assert int(f_resp.headers.get("Content-Length", 0)) > 0
        else:
            assert not rep["file_url"], f"Report {rep['id']} without document should have empty file_url"
    print(f"PASS: Report documents validated ({doc_count} reports have verified disk documents, zero 404 links)")

if __name__ == "__main__":
    test_lab_report_table()
    test_lab_report_trends()
    test_lab_report_documents_and_urls()
    print("\nALL LAB REPORT BUG VERIFICATIONS PASSED SUCCESSFULLY!")
