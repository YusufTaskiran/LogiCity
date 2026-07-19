import argparse
import pickle as pkl


def build_parser():
    parser = argparse.ArgumentParser(description="Print per-step agent coordinates from a saved eval trace.")
    parser.add_argument("--trace", required=True, help="Path to <exp>_<episode>_trace.pkl")
    parser.add_argument(
        "--agent-names",
        default="Car_1,Car_2",
        help="Comma-separated agent names to print.",
    )
    return parser


def index_agents(agent_list):
    return {agent["name"]: agent for agent in agent_list}


def main():
    args = build_parser().parse_args()
    target_names = [name.strip() for name in args.agent_names.split(",") if name.strip()]

    with open(args.trace, "rb") as f:
        trace = pkl.load(f)

    print(f"episode_id={trace.get('episode_id')}")
    print(f"success={trace.get('success')} timeout={trace.get('timeout')} final_score={trace.get('final_score')}")
    print()

    for step in trace["steps"]:
        pre_agents = index_agents(step.get("pre_agents", []))
        post_agents = index_agents(step.get("post_agents", []))
        chosen_action = step.get("chosen_action")
        print(f"step={step['step']} action={chosen_action} reward={step.get('reward')} done={step.get('done')}")
        for name in target_names:
            pre = pre_agents.get(name)
            post = post_agents.get(name)
            if pre is None and post is None:
                continue
            pre_pos = pre.get("pos") if pre is not None else None
            post_pos = post.get("pos") if post is not None else None
            reached = post.get("reach_goal") if post is not None else None
            print(f"  {name}: {pre_pos} -> {post_pos} reach_goal={reached}")
        print()


if __name__ == "__main__":
    main()
