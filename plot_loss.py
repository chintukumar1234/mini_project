import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss():
    """
    Training के actual losses को graph में plot करता है
    (Terminal से नोट किए गए losses से)
    """
    
    # Training से actual losses (50 epochs)
    train_losses = [
        2.7411646842956543, 2.676898717880249, 2.6322882175445557, 2.6008646488189697, 2.5746805667877197,
        2.549555540084839, 2.5240790843963623, 2.500741481781006, 2.482893466949463, 2.4655795097351074,
        2.4481213092803955, 2.4313037395477295, 2.415895700454712, 2.4002811908721924, 2.3856239318847656,
        2.3703601360321045, 2.3540756702423096, 2.337848424911499, 2.322190046310425, 2.3080878257751465,
        2.2944135665893555, 2.281595230102539, 2.2700252532958984, 2.259044647216797, 2.2486467361450195,
        2.238999605178833, 2.2300353050231934, 2.2220187187194824, 2.21435809135437, 2.2074549198150635,
        2.200953483581543, 2.194546699523926, 2.188896417617798, 2.184126377105713, 2.1801857948303223,
        2.174955129623413, 2.1711606979370117, 2.1669259071350098, 2.163337469100952, 2.1593759059906006,
        2.1566219329833984, 2.1531355381011963, 2.1501176357269287, 2.146827459335327, 2.144624948501587,
        2.1419403553009033, 2.1390879154205322, 2.136667013168335, 2.1340513229370117, 2.1318020820617676
    ]
    
    val_losses = [
        2.4406135082244873, 2.4388928413391113, 2.4361109733581543, 2.4333279132843018, 2.4297451972961426,
        2.425466299057007, 2.421140193939209, 2.4170846939086914, 2.4132683277130127, 2.4087979793548584,
        2.4030046463012695, 2.396218776702881, 2.3892927169799805, 2.382249593734741, 2.375065326690674,
        2.3681371212005615, 2.3608686923980713, 2.352271318435669, 2.3434665203094482, 2.3359663486480713,
        2.32924747467041, 2.326171636581421, 2.326032876968384, 2.3278303146362305, 2.3267862796783447,
        2.32096791267395, 2.32039475440979, 2.294214963912964, 2.2805111408233643, 2.2693064212799072,
        2.25860595703125, 2.2519662380218506, 2.241689682006836, 2.2359626293182373, 2.211280107498169,
        2.189512252807617, 2.1814799308776855, 2.1803908348083496, 2.175034761428833, 2.1667885780334473,
        2.1672303676605225, 2.1646671295166016, 2.1583707332611084, 2.1546897888183594, 2.154987335205078,
        2.1502270698547363, 2.1449193954467773, 2.1428089141845703, 2.142699956893921, 2.136458396911621
    ]
    
    print("📊 Plotting training history...")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, 51)
    
    ax.plot(epochs, train_losses, label="Training Loss", linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_losses, label="Validation Loss", linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
    ax.set_ylabel("Loss", fontsize=12, fontweight='bold')
    ax.set_title("Training History - Loss Over 50 Epochs", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add some statistics
    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    min_train_epoch = train_losses.index(min_train_loss) + 1
    min_val_epoch = val_losses.index(min_val_loss) + 1
    
    textstr = f'Min Train Loss: {min_train_loss:.4f} (Epoch {min_train_epoch})\nMin Val Loss: {min_val_loss:.4f} (Epoch {min_val_epoch})'
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches='tight')
    print("✅ Loss graph saved as 'training_history.png'")
    
    # Print statistics
    print(f"\n📈 Training Statistics:")
    print(f"   Initial Training Loss: {train_losses[0]:.4f}")
    print(f"   Final Training Loss: {train_losses[-1]:.4f}")
    print(f"   Improvement: {train_losses[0] - train_losses[-1]:.4f} ({(1 - train_losses[-1]/train_losses[0])*100:.1f}%)")
    print(f"\n   Initial Validation Loss: {val_losses[0]:.4f}")
    print(f"   Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"   Improvement: {val_losses[0] - val_losses[-1]:.4f} ({(1 - val_losses[-1]/val_losses[0])*100:.1f}%)")
    print(f"\n   Best Model Epoch: {min_val_epoch} (Validation Loss: {min_val_loss:.4f})")
    
    plt.show()


if __name__ == "__main__":
    plot_training_loss()
