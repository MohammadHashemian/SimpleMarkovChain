from domain.enums import ArthropathySeverity


def pettersson_to_severity(score: int) -> ArthropathySeverity:
    if not (0 <= score <= 79):
        raise ValueError("Pettersson score must be in range [0, 79]")

    if score == 0:
        return ArthropathySeverity.HEALTHY
    elif score <= 4:
        return ArthropathySeverity.MILD
    elif score <= 27:
        return ArthropathySeverity.MODERATE
    else:
        return ArthropathySeverity.SEVERE
