import { SCENARIOS, type Scenario } from "../data/scenarios";

type Props = {
  disabled?: boolean;
  onSelect: (scenario: Scenario) => void;
};

export function ScenarioChips({ disabled, onSelect }: Props) {
  return (
    <div className="scenarios">
      <p className="scenarios-label">Demo scenarios</p>
      <div className="scenario-row">
        {SCENARIOS.map((s) => (
          <button
            key={s.id}
            type="button"
            className="scenario-chip"
            disabled={disabled}
            title={s.description}
            onClick={() => onSelect(s)}
          >
            <span className="scenario-title">{s.label}</span>
            <span className="scenario-desc">{s.description}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
