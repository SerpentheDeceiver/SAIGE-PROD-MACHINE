def route_query(query: str) -> str:
    query = query.lower()

    syllabus_keywords = [
        "syllabus", "unit", "module", "course code",
        "semester", "topics", "learning outcome",
        "subject", "lab", "credits per course"
    ]
    admin_keywords = [
        "regulation", "grading", "attendance", "exam",
        "eligibility", "cgpa", "credits required",
        "backlog", "passing criteria"
    ]
    institutional_keywords = [
        "admission", "fee", "hostel", "scholarship",
        "portal", "login", "placement", "library",
        "contact", "faculty", "office"
    ]

    for word in syllabus_keywords:
        if word in query:
            return "syllabus"
    for word in admin_keywords:
        if word in query:
            return "admin"
    for word in institutional_keywords:
        if word in query:
            return "institutional"

    return "unknown"
