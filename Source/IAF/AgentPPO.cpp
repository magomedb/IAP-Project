// Fill out your copyright notice in the Description page of Project Settings.


#include "AgentPPO.h"
//#include <torch/torch.h>
#include <tensorflow/c/c_api.h>

// Sets default values for this component's properties
UAgentPPO::UAgentPPO()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


void UAgentPPO::MyTestFunction(FString SomeString)
{
	//torch::Tensor tensor = torch::rand({ 2, 3 });
	FString SomeString2 = TF_Version();
	UE_LOG(LogTemp, Log, TEXT("You wanted to say %s"), *SomeString2);
}

// Called when the game starts
void UAgentPPO::BeginPlay()
{
	Super::BeginPlay();

	// ...
	
}


// Called every frame
void UAgentPPO::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}

