from typing import Any


def artifact_sorter(
        artifacts: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
    return sorted(artifacts, key=lambda a: a['power'], reverse=True)


def power_filter(
        mage: list[dict[str, Any]],
        min_power: int
        ) -> list[dict[str, Any]]:
    return list(filter(lambda x: x['power'] >= min_power, mage))


def spell_transformer(spell: list[str]) -> list[str]:
    return list(map(lambda x: "* " + x + " *", spell))


def mage_stats(
        m: list[dict[str, Any]]
        ) -> dict[str, Any]:
    return {'max_power': max(m, key=lambda x: x['power'])['power'],
            'min_power': min(m, key=lambda y: y['power'])['power'],
            'avg_power': round(sum(map(lambda m: m['power'], m)) / len(m), 2)}


if __name__ == "__main__":
    artifacts = [
        {'name': 'Wind Cloak', 'power': 81, 'type': 'weapon'},
        {'name': 'Fire Staff', 'power': 80, 'type': 'armor'},
        {'name': 'Water Chalice', 'power': 91, 'type': 'weapon'},
        {'name': 'Earth Shield', 'power': 114, 'type': 'accessory'}
        ]
    mages = [
        {'name': 'Nova', 'power': 56, 'element': 'ice'},
        {'name': 'River', 'power': 78, 'element': 'wind'},
        {'name': 'Ash', 'power': 95, 'element': 'shadow'},
        {'name': 'Morgan', 'power': 75, 'element': 'water'},
        {'name': 'Riley', 'power': 51, 'element': 'ice'}
        ]
    spells = ['tornado', 'heal', 'blizzard', 'meteor']
    print(artifact_sorter(artifacts))
    print(power_filter(mages, 55))
    print(spell_transformer(spells))
    print(mage_stats(mages))
